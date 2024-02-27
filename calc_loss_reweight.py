def _calc_loss_rescales(
    self, speech: torch.Tensor, speech_lengths: torch.Tensor,
    aux_speech: torch.Tensor, aux_speech_lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Calculate loss

    Args:
        speech: (Batch, Length, Dim)
        speech_lengths: (Batch, )
        aux_speech: (Batch, Length, Dim)
        aux_speech_lengths: (Batch, )
    """
    with autocast(False):
        # 1. Extract feats
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        aux_feats, aux_feats_lengths = self._extract_feats(aux_speech, aux_speech_lengths)

    def calc_vqvae_avg_probs(xs, xs_lens):
        vae_inputs = xs[:, :, :self.rescaler_model_conf["VAE_n_lower_fre_bands"]]
        h_vae_enc, h_vae_enc_mask = self.vae_encoder(vae_inputs, xs_lens)
        _, h_vq, indices = self.vq_embedding.encode(h_vae_enc)
        bb, tt, dd = h_vq.size()
        indices = torch.reshape(indices, (bb, tt))
        encodings = F.one_hot(indices, self.vq_embedding.embedding.size(0)).float()
        mask = torch.transpose(h_vae_enc_mask, 1, 2)  # -> B x T x 1
        rep = torch.sum(encodings * mask, dim=1) / torch.sum(mask, dim=1)
        return rep

    def calc_vqvae_code_dis(xs, xs_lens):
        vae_inputs = xs[:, :, :self.rescaler_model_conf["VAE_n_lower_fre_bands"]]
        h_vae_enc, h_vae_enc_mask = self.vae_encoder(vae_inputs, xs_lens)
        _, h_vq, indices, _code_dis = self.vq_embedding.encode(h_vae_enc)
        bb, tt, dd = h_vq.size()
        _code_dis = torch.reshape(_code_dis, (bb, tt, self.vq_embedding.n_embeddings))  # -> B x T x N_code
        code_mask = torch.transpose(h_vae_enc_mask, 1, 2)  # -> B x T x 1
        return _code_dis, code_mask

    reps = calc_vqvae_avg_probs(feats, feats_lengths)
    aux_reps = calc_vqvae_avg_probs(aux_feats, aux_feats_lengths)
    # scatter representations to calculate Cartesian product
    reps = torch.unsqueeze(reps, 1)  # B, 1, D
    aux_reps = torch.unsqueeze(aux_reps, 0)  # 1, B, D
    similarity = F.cosine_similarity(reps, aux_reps, dim=2, eps=1e-8)
    raw_loss_weights = torch.mean(similarity, dim=1)

    if "threshold" in self.rescaler_model_conf and self.rescaler_model_conf["threshold"] is not None:
        mask = (raw_loss_weights > self.rescaler_model_conf["threshold"]).float()
        raw_loss_weights = mask * raw_loss_weights + (1.0 - mask) * 0.1 * raw_loss_weights

    # rescale the summation of loss weights to batch size
    batch_size = feats.size(0)
    #loss_weights = raw_loss_weights + 1
    pat_loss_weights = raw_loss_weights * (batch_size / torch.sum(raw_loss_weights))
    #loss_weights = (loss_weights - loss_weights.min()) / (loss_weights.max() - loss_weights.min())
    # loss_weights = torch.clamp(raw_loss_weights, min=0, max=self.simi_max_value) / self.simi_max_value
    similarity = F.sigmoid(
        (raw_loss_weights - self.simi_max_value / 2) * (10.0 / self.simi_max_value)
    )
    return similarity.detach(), pat_loss_weights.detach()

# same in espnet, 80-dim fbank feature
def _extract_feats(
    self, speech: torch.Tensor, speech_lengths: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert speech_lengths.dim() == 1, speech_lengths.shape

    # for data-parallel
    speech = speech[:, : speech_lengths.max()]

    if self.frontend is not None:
        # Frontend
        #  e.g. STFT and Feature extract
        #       data_loader may send time-domain signal in this case
        # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
        feats, feats_lengths = self.frontend(speech, speech_lengths)
    else:
        # No frontend and no feature extract
        feats, feats_lengths = speech, speech_lengths
    return feats, feats_lengths