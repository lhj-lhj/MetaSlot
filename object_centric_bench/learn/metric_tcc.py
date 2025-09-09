from einops import repeat
import torch as pt
import torch.nn.functional as ptnf

# from tcc.alignment import compute_alignment_loss


class TCC:

    def __init__(
        self,
        batch_size,
        stochastic_matching=False,
        normalize_embeddings=False,
        loss_type="classification",
        similarity_type="l2",
        num_cycles=20,
        cycle_length=2,
        temperature=0.1,
        label_smoothing=0.1,
        variance_lambda=0.001,
        huber_delta=0.1,
        normalize_indices=True,
    ):
        self.batch_size = batch_size
        self.stochastic_matching = stochastic_matching
        self.normalize_embeddings = normalize_embeddings
        self.loss_type = loss_type
        self.similarity_type = similarity_type
        self.num_cycles = num_cycles
        self.cycle_length = cycle_length
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.variance_lambda = variance_lambda
        self.huber_delta = huber_delta
        self.normalize_indices = normalize_indices

    def __call__(self, input, steps=None, seq_lens=None):
        """
        input: frame-level embeddings in shape (b,t,c)
        """
        return compute_alignment_loss(
            input,
            self.batch_size,
            steps,
            seq_lens,
            stochastic_matching=self.stochastic_matching,
            normalize_embeddings=self.normalize_embeddings,
            loss_type=self.loss_type,
            similarity_type=self.similarity_type,
            num_cycles=self.num_cycles,
            cycle_length=self.cycle_length,
            temperature=self.temperature,
            label_smoothing=self.label_smoothing,
            variance_lambda=self.variance_lambda,
            huber_delta=self.huber_delta,
            normalize_indices=self.normalize_indices,
        )


def compute_alignment_loss(
    embs,
    batch_size,
    steps=None,
    seq_lens=None,
    stochastic_matching=False,
    normalize_embeddings=False,
    loss_type="classification",
    similarity_type="l2",
    num_cycles=20,
    cycle_length=2,
    temperature=0.1,
    label_smoothing=0.1,
    variance_lambda=0.001,
    huber_delta=0.1,
    normalize_indices=True,
):
    """Computes alignment loss between sequences of embeddings.

    There are four major hparams that need to be tuned while applying the loss:
    i) Should the loss be applied with L2 normalization on the embeddings or
    without it?
    ii) Should we perform stochastic alignment of sequences? This means should we
    use all the steps of the embedding or only choose a random subset for
    alignment?
    iii) Should we apply cycle-consistency constraints using a classification loss
    or a regression loss? (Section 3 in paper)
    iv) Should the similarity metric be based on an L2 distance or cosine
    similarity?

    Other hparams that can be used to control how hard/soft we want the alignment
    between different sequences to be:
    i) temperature (all losses)
    ii) label_smoothing (classification)
    iii) variance_lambda (regression_mse_var)
    iv) huber_delta (regression_huber)
    Each of these params are used in their respective loss types (in brackets) and
    allow the application of the cycle-consistency constraints in a controllable
    manner but they do so in very different ways. Please refer to paper for more
    details.

    The default hparams work well for frame embeddings of videos of humans
    performing actions. Other datasets might need different values of hparams.

    Args:
      embs: Tensor, sequential embeddings of the shape [N, T, D] where N is the
        batch size, T is the number of timesteps in the sequence, D is the size of
        the embeddings.
      batch_size: Integer, Size of the batch.
      steps: Tensor, step indices/frame indices of the embeddings of the shape
        [N, T] where N is the batch size, T is the number of the timesteps.
        If this is set to None, then we assume that the sampling was done in a
        uniform way and use tf.range(num_steps) as the steps.
      seq_lens: Tensor, Lengths of the sequences from which the sampling was done.
        This can provide additional information to the alignment loss. This is
        different from num_steps which is just the number of steps that have been
        sampled from the entire sequence.
      stochastic_matching: Boolean, Should the used for matching be sampled
        stochastically or deterministically? Deterministic is better for TPU.
        Stochastic is better for adding more randomness to the training process
        and handling long sequences.
      normalize_embeddings: Boolean, Should the embeddings be normalized or not?
        Default is to use raw embeddings. Be careful if you are normalizing the
        embeddings before calling this function.
      loss_type: String, This specifies the kind of loss function to use.
        Currently supported loss functions: classification, regression_mse,
        regression_mse_var, regression_huber.
      similarity_type: String, Currently supported similarity metrics: l2, cosine.
      num_cycles: Integer, number of cycles to match while aligning
        stochastically.  Only used in the stochastic version.
      cycle_length: Integer, Lengths of the cycle to use for matching. Only used
        in the stochastic version. By default, this is set to 2.
      temperature: Float, temperature scaling used to scale the similarity
        distributions calculated using the softmax function.
      label_smoothing: Float, Label smoothing argument used in
        tf.keras.losses.categorical_crossentropy function and described in this
        paper https://arxiv.org/pdf/1701.06548.pdf.
      variance_lambda: Float, Weight of the variance of the similarity
        predictions while cycling back. If this is high then the low variance
        similarities are preferred by the loss while making this term low results
        in high variance of the similarities (more uniform/random matching).
      huber_delta: float, Huber delta described in tf.keras.losses.huber_loss.
      normalize_indices: Boolean, If True, normalizes indices by sequence lengths.
        Useful for ensuring numerical instabilities doesn't arise as sequence
        indices can be large numbers.

    Returns:
      loss: Tensor, Scalar loss tensor that imposes the chosen variant of the
        cycle-consistency loss.
    """
    device = embs.device
    _, num_steps, c = embs.shape
    if steps is None:
        steps = repeat(
            pt.arange(num_steps, device=device), "(1 c) -> b c", b=batch_size
        )
    if seq_lens is None:
        seq_lens = pt.from_numpy(np.array(num_steps, "int64"), device=device)[
            None
        ].expand(batch_size)
    if num_cycles is None:  # XXX
        num_cycles = batch_size * num_steps
    if normalize_embeddings:
        embs = embs / pt.norm(embs, dim=2, keepdim=True)
    if stochastic_matching:
        loss = compute_stochastic_alignment_loss(
            embs=embs,
            steps=steps,
            seq_lens=seq_lens,
            num_steps=num_steps,
            batch_size=batch_size,
            loss_type=loss_type,
            similarity_type=similarity_type,
            num_cycles=num_cycles,
            cycle_length=cycle_length,
            temperature=temperature,
            label_smoothing=label_smoothing,
            variance_lambda=variance_lambda,
            huber_delta=huber_delta,
            normalize_indices=normalize_indices,
        )
    else:
        loss = compute_deterministic_alignment_loss(
            embs=embs,
            steps=steps,
            seq_lens=seq_lens,
            num_steps=num_steps,
            batch_size=batch_size,
            loss_type=loss_type,
            similarity_type=similarity_type,
            temperature=temperature,
            label_smoothing=label_smoothing,
            variance_lambda=variance_lambda,
            huber_delta=huber_delta,
            normalize_indices=normalize_indices,
        )
    return loss


def compute_stochastic_alignment_loss(
    embs,
    steps,
    seq_lens,
    num_steps,
    batch_size,
    loss_type,
    similarity_type,
    num_cycles,
    cycle_length,
    temperature,
    label_smoothing,
    variance_lambda,
    huber_delta,
    normalize_indices,
):
    """Compute cycle-consistency loss by stochastically sampling cycles.

    Args:
      embs: Tensor, sequential embeddings of the shape [N, T, D] where N is the
        batch size, T is the number of timesteps in the sequence, D is the size of
        the embeddings.
      steps: Tensor, step indices/frame indices of the embeddings of the shape
        [N, T] where N is the batch size, T is the number of the timesteps.
      seq_lens: Tensor, Lengths of the sequences from which the sampling was done.
        This can provide additional information to the alignment loss.
      num_steps: Integer/Tensor, Number of timesteps in the embeddings.
      batch_size: Integer/Tensor, Batch size.
      loss_type: String, This specifies the kind of loss function to use.
        Currently supported loss functions: 'classification', 'regression_mse',
        'regression_mse_var', 'regression_huber'.
      similarity_type: String, Currently supported similarity metrics: 'l2',
        'cosine'.
      num_cycles: Integer, number of cycles to match while aligning
        stochastically.  Only used in the stochastic version.
      cycle_length: Integer, Lengths of the cycle to use for matching. Only used
        in the stochastic version. By default, this is set to 2.
      temperature: Float, temperature scaling used to scale the similarity
        distributions calculated using the softmax function.
      label_smoothing: Float, Label smoothing argument used in
        tf.keras.losses.categorical_crossentropy function and described in this
        paper https://arxiv.org/pdf/1701.06548.pdf.
      variance_lambda: Float, Weight of the variance of the similarity
        predictions while cycling back. If this is high then the low variance
        similarities are preferred by the loss while making this term low results
        in high variance of the similarities (more uniform/random matching).
      huber_delta: float, Huber delta described in tf.keras.losses.huber_loss.
      normalize_indices: Boolean, If True, normalizes indices by sequence lengths.
        Useful for ensuring numerical instabilities doesn't arise as sequence
        indices can be large numbers.

    Returns:
      loss: Tensor, Scalar loss tensor that imposes the chosen variant of the
        cycle-consistency loss.
    """
    cycles = gen_cycles(num_cycles, batch_size, cycle_length).to(embs.device)
    logits, labels = _align(
        cycles, embs, num_steps, num_cycles, cycle_length, similarity_type, temperature
    )  # (b,t), (b,)
    labels = labels.detach()
    steps = steps.detach()

    if loss_type == "classification":
        loss = ptnf.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    elif "regression" in loss_type:
        labels = ptnf.one_hot(labels, embs.size(1))
        steps = steps.gather(0, cycles[:, 0])
        seq_lens = seq_lens.gather(0, cycles[:, 0])
        loss = cycle_back_regression_loss(
            logits,
            labels,
            num_steps,
            steps,
            seq_lens,
            loss_type,
            normalize_indices,
            variance_lambda,
            huber_delta,
        )
    else:
        raise "ValueError"
    return loss


def compute_deterministic_alignment_loss(
    embs,
    steps,
    seq_lens,
    num_steps,
    batch_size,
    loss_type,
    similarity_type,
    temperature,
    label_smoothing,
    variance_lambda,
    huber_delta,
    normalize_indices,
):
    """Compute cycle-consistency loss for all steps in each sequence.

    This aligns each pair of videos in the batch except with itself.
    When aligning it also matters which video is the starting video. So for N
    videos in the batch, we have N * (N-1) alignments happening.
    For example, a batch of size 3 has 6 pairs of sequence alignments.


    Args:
      embs: Tensor, sequential embeddings of the shape [N, T, D] where N is the
        batch size, T is the number of timesteps in the sequence, D is the size
        of the embeddings.
      steps: Tensor, step indices/frame indices of the embeddings of the shape
        [N, T] where N is the batch size, T is the number of the timesteps.
      seq_lens: Tensor, Lengths of the sequences from which the sampling was
      done. This can provide additional information to the alignment loss.
      num_steps: Integer/Tensor, Number of timesteps in the embeddings.
      batch_size: Integer, Size of the batch.
      loss_type: String, This specifies the kind of loss function to use.
        Currently supported loss functions: 'classification', 'regression_mse',
        'regression_mse_var', 'regression_huber'.
      similarity_type: String, Currently supported similarity metrics: 'l2' ,
        'cosine' .
      temperature: Float, temperature scaling used to scale the similarity
        distributions calculated using the softmax function.
      label_smoothing: Float, Label smoothing argument used in
        tf.keras.losses.categorical_crossentropy function and described in this
        paper https://arxiv.org/pdf/1701.06548.pdf.
      variance_lambda: Float, Weight of the variance of the similarity
        predictions while cycling back. If this is high then the low variance
        similarities are preferred by the loss while making this term low
        results in high variance of the similarities (more uniform/random
        matching).
      huber_delta: float, Huber delta described in tf.keras.losses.huber_loss.
      normalize_indices: Boolean, If True, normalizes indices by sequence
        lengths. Useful for ensuring numerical instabilities doesn't arise as
        sequence indices can be large numbers.
    Returns:
      loss: Tensor, Scalar loss tensor that imposes the chosen variant of the
          cycle-consistency loss.
    """
    labels_list = []
    logits_list = []
    steps_list = []
    seq_lens_list = []

    for i in range(batch_size):
        for j in range(batch_size):
            # We do not align the sequence with itself.
            if i != j:
                logits, labels = align_pair_of_sequences(
                    embs[i], embs[j], similarity_type, temperature
                )
                logits_list.append(logits)
                labels_list.append(labels)
                steps_list.append(pt.tile(steps[i : i + 1], [num_steps, 1]))
                seq_lens_list.append(pt.tile(seq_lens[i : i + 1], [num_steps]))

    logits = pt.concat(logits_list, dim=0)
    labels = pt.concat(labels_list, dim=0)
    steps = pt.concat(steps_list, dim=0)
    seq_lens = pt.concat(seq_lens_list, dim=0)

    if loss_type == "classification":
        loss = classification_loss(logits, labels, label_smoothing)
    elif "regression" in loss_type:
        loss = cycle_back_regression_loss(
            logits,
            labels,
            num_steps,
            steps,
            seq_lens,
            loss_type,
            normalize_indices,
            variance_lambda,
            huber_delta,
        )
    else:
        raise "ValueError"

    return loss


def pairwise_l2_distance(embs1, embs2):  # TODO pt.norm()
    """Computes pairwise distances between all rows of embs1 and embs2."""
    norm1 = embs1.square().sum(1).view(-1, 1)
    norm2 = embs2.square().sum(1).view(1, -1)
    # Max to ensure matmul doesn't produce anything negative due to floating
    # point approximations.
    dist = (norm1 + norm2 - 2.0 * pt.einsum("ab,cb->ac", embs1, embs2)).clip(0.0)
    return dist


def get_scaled_similarity(embs1, embs2, similarity_type, temperature):
    """Returns similarity between each all rows of embs1 and all rows of embs2.

    The similarity is scaled by the number of channels/embedding size and
    temperature.

    Args:
      embs1: Tensor, Embeddings of the shape [M, D] where M is the number of
        embeddings and D is the embedding size.
      embs2: Tensor, Embeddings of the shape [N, D] where N is the number of
        embeddings and D is the embedding size.
      similarity_type: String, Either one of 'l2' or 'cosine'.
      temperature: Float, Temperature used in scaling logits before softmax.

    Returns:
      similarity: Tensor, [M, N] tensor denoting similarity between embs1 and
        embs2.
    """
    channels = embs1.shape[1]  # tf.cast(tf.shape(embs1)[1], tf.float32)
    # Go for embs1 to embs2.
    if similarity_type == "cosine":
        similarity = pt.einsum("ab,cb->ac", embs1, embs2)
    elif similarity_type == "l2":  # TODO XXX pt.norm()
        similarity = -1.0 * pairwise_l2_distance(embs1, embs2)
    else:
        raise ValueError("similarity_type can either be l2 or cosine.")

    # Scale the distance  by number of channels. This normalization helps with optimization.
    similarity /= channels
    # Scale the distance by a temperature that helps with how soft/hard the alignment should be.
    similarity /= temperature

    return similarity


def align_pair_of_sequences(embs1, embs2, similarity_type, temperature):
    """Align a given pair embedding sequences.

    Args:
      embs1: Tensor, Embeddings of the shape [M, D] where M is the number of
        embeddings and D is the embedding size.
      embs2: Tensor, Embeddings of the shape [N, D] where N is the number of
        embeddings and D is the embedding size.
      similarity_type: String, Either one of 'l2' or 'cosine'.
      temperature: Float, Temperature used in scaling logits before softmax.
    Returns:
       logits: Tensor, Pre-softmax similarity scores after cycling back to the
        starting sequence.
      labels: Tensor, One hot labels containing the ground truth. The index where
        the cycle started is 1.
    """
    device = embs1.device
    max_num_steps = embs1.shape[0]

    # Find distances between embs1 and embs2.
    sim_12 = get_scaled_similarity(embs1, embs2, similarity_type, temperature)
    # Softmax the distance.
    softmaxed_sim_12 = sim_12.softmax(1)

    # Calculate soft-nearest neighbors.
    nn_embs = pt.einsum("ab,bc->ac", softmaxed_sim_12, embs2)  # TODO XXX ???

    # Find distances between nn_embs and embs1.
    sim_21 = get_scaled_similarity(nn_embs, embs1, similarity_type, temperature)

    logits = sim_21
    labels = ptnf.one_hot(pt.arange(max_num_steps, device=device), max_num_steps)

    return logits, labels


def _align_single_cycle(
    cycle, embs, cycle_length, num_steps, similarity_type, temperature
):
    """Takes a single cycle and returns logits (simialrity scores) and labels."""
    device = embs.device
    batch_size, num_steps, num_channels = embs.shape
    n_idx = pt.randint(0, num_steps, [], device=device)

    query_feats = embs[cycle[0], n_idx : n_idx + 1]  # (t=1,c)

    for c in range(1, cycle_length + 1):
        candidate_feats = embs[cycle[c], :, :]  # (t,c)

        if similarity_type == "l2":
            mse = ptnf.mse_loss(
                query_feats.repeat([num_steps, 1]), candidate_feats, reduction="none"
            )  # (t,c)
            similarity = -mse.sum(1)  # (t,)
        elif similarity_type == "cosine":
            similarity = pt.einsum("tc,c->t", candidate_feats, query_feats[0, :])
        else:
            raise "ValueError"

        similarity = similarity / num_channels / temperature  # (t,)
        weight = similarity.softmax(0)
        query_feats = pt.einsum("tc,t->c", candidate_feats, weight)[None, :]  # (t=1,c)

    return similarity, n_idx


def _align(
    cycles, embs, num_steps, num_cycles, cycle_length, similarity_type, temperature
):
    """Align by finding cycles in embs."""
    logits = []
    labels = []
    for i in range(num_cycles):
        logit, label = _align_single_cycle(
            cycles[i], embs, cycle_length, num_steps, similarity_type, temperature
        )
        logits.append(logit)
        labels.append(label)
    logits = pt.stack(logits)
    labels = pt.stack(labels)
    return logits, labels


def gen_cycles(num_cycles, batch_size, cycle_length=2):
    """Generates cycles for alignment.

    Generates a batch of indices to cycle over. For example setting num_cycles=2,
    batch_size=5, cycle_length=3 might return something like this:
    cycles = [[0, 3, 4, 0], [1, 2, 0, 1]]. This means we have 2 cycles for which
    the loss will be calculated. The first cycle starts at sequence 0 of the
    batch, then we find a matching step in sequence 3 of that batch, then we
    find matching step in sequence 4 and finally come back to sequence 0,
    completing a cycle.

    Args:
      num_cycles: Integer, Number of cycles that will be matched in one pass.
      batch_size: Integer, Number of sequences in one batch.
      cycle_length: Integer, Length of the cycles. If we are matching between
        2 sequences (cycle_length=2), we get cycles that look like [0,1,0].
        This means that we go from sequence 0 to sequence 1 then back to sequence
        0. A cycle length of 3 might look like [0, 1, 2, 0].

    Returns:
      cycles: Tensor, Batch indices denoting cycles that will be used for
        calculating the alignment loss.
    """
    cycles = pt.stack([pt.randperm(batch_size) for _ in range(num_cycles)], 0)
    cycles = pt.cat([cycles[:, :cycle_length], cycles[:, :1]], 1)
    return cycles


def classification_loss(logits, labels, label_smoothing):
    """Loss function based on classifying the correct indices.

    In the paper, this is called Cycle-back Classification.

    Args:
      logits: Tensor, Pre-softmax scores used for classification loss. These are
        similarity scores after cycling back to the starting sequence.
      labels: Tensor, One hot labels containing the ground truth. The index where
        the cycle started is 1.
      label_smoothing: Float, label smoothing factor which can be used to
        determine how hard the alignment should be.
    Returns:
      loss: Tensor, A scalar classification loss calculated using standard softmax
        cross-entropy loss.
    """
    # Just to be safe, we stop gradients from labels as we are generating labels.
    assert logits.ndim == 2 and labels.ndim == 1  # XXX GeneralZ
    return ptnf.cross_entropy(logits, labels.detach(), label_smoothing=label_smoothing)


def cycle_back_regression_loss(
    logits,
    labels,
    num_steps,
    steps,
    seq_lens,
    loss_type,
    normalize_indices,
    variance_lambda,
    huber_delta,
):
    """Loss function based on regressing to the correct indices.

    In the paper, this is called Cycle-back Regression. There are 3 variants
    of this loss:
    i) regression_mse: MSE of the predicted indices and ground truth indices.
    ii) regression_mse_var: MSE of the predicted indices that takes into account
    the variance of the similarities. This is important when the rate at which
    sequences go through different phases changes a lot. The variance scaling
    allows dynamic weighting of the MSE loss based on the similarities.
    iii) regression_huber: Huber loss between the predicted indices and ground
    truth indices.


    Args:
      logits: Tensor, Pre-softmax similarity scores after cycling back to the
        starting sequence.
      labels: Tensor, One hot labels containing the ground truth. The index where
        the cycle started is 1.
      num_steps: Integer, Number of steps in the sequence embeddings.
      steps: Tensor, step indices/frame indices of the embeddings of the shape
        [N, T] where N is the batch size, T is the number of the timesteps.
      seq_lens: Tensor, Lengths of the sequences from which the sampling was done.
        This can provide additional temporal information to the alignment loss.
      loss_type: String, This specifies the kind of regression loss function.
        Currently supported loss functions: regression_mse, regression_mse_var,
        regression_huber.
      normalize_indices: Boolean, If True, normalizes indices by sequence lengths.
        Useful for ensuring numerical instabilities don't arise as sequence
        indices can be large numbers.
      variance_lambda: Float, Weight of the variance of the similarity
        predictions while cycling back. If this is high then the low variance
        similarities are preferred by the loss while making this term low results
        in high variance of the similarities (more uniform/random matching).
      huber_delta: float, Huber delta described in tf.keras.losses.huber_loss.

    Returns:
       loss: Tensor, A scalar loss calculated using a variant of regression.
    """
    if normalize_indices:
        steps = steps.float() / seq_lens[:, None].float()
    else:
        steps = steps.float()
    weight = logits.softmax(-1)
    true_time = pt.einsum("bt,bt->b", steps, labels.float())
    pred_time = pt.einsum("bt,bt->b", steps, weight)

    if loss_type in ["regression_mse", "regression_mse_var"]:
        if "var" in loss_type:  # Variance aware regression
            pred_time_variance = ((steps - pred_time[:, None]) ** 2 * weight).sum(1)
            pred_time_log_var = pred_time_variance.log()
            return (
                (-pred_time_log_var).exp() * (true_time - pred_time) ** 2
                + variance_lambda * pred_time_log_var
            ).mean()
        else:
            return ptnf.mse_loss(pred_time, true_time)
    elif loss_type == "regression_huber":
        return ptnf.huber_loss(pred_time, true_time, delta=huber_delta)
    else:
        raise "ValueError"
