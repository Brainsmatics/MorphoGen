import torch
import random
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def reconstruction_loss(reconstructed_x, x, ignore_element=0):
    # reconstruction loss
    # x = [trg len, batch size * n walks, output dim] when tree major
    # x = [trg len, batch size, output dim] when batch major

    seq_len, batch_size, output_dim = x.shape
    mask = x[:, :, 0] != ignore_element
    rec_loss = 0
    # print(torch.all(mask != torch.isinf(x[:, :, 0])))
    for d in range(output_dim):
        # print(reconstructed_x[:, :, d][mask])
        # print(x[:, :, d][mask])
        rec_loss += torch.nn.functional.mse_loss(
            reconstructed_x[:, :, d][mask],
            x[:, :, d][mask], reduction='sum'
        )
        # print(rec_loss)
    return rec_loss / output_dim

class ConditionalSeqEncoder(torch.nn.Module):
    # branch encoder
    # encode one branch into states of
    # hidden & cell (both [n_layers,hidden_dim])
    # Same as the SeqEncoder
    def __init__(
            self, input_dim, embedding_dim,
            hidden_dim, n_layers=2, dropout=0.5
    ):
        super(ConditionalSeqEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.coordinate2emb = torch.nn.Linear(input_dim, embedding_dim)
        self.dropout_fun = torch.nn.Dropout(dropout)
        self.rnn = torch.nn.LSTM(
            embedding_dim, hidden_dim, n_layers, dropout=dropout
        )

    def forward(self, src, seq_len):
        input_seq = self.dropout_fun(self.coordinate2emb(src))
        packed_seq = pack_padded_sequence(
            input_seq, seq_len, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.rnn(packed_seq)

        # outputs = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        return hidden, cell


class ConditionalSeqDecoder(torch.nn.Module):
    # Same as the SeqDecoder
    def __init__(
            self, output_dim, embedding_dim, hidden_dim,
            n_layers=2, dropout=0.5
    ):
        super(ConditionalSeqDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.hid2coordinate = torch.nn.Linear(hidden_dim, output_dim)
        self.coordinate2emb = torch.nn.Linear(output_dim, embedding_dim)
        self.dropout_fun = torch.nn.Dropout(dropout)
        self.rnn = torch.nn.LSTM(
            embedding_dim, hidden_dim, n_layers, dropout=dropout
        )

    def forward(self, init, hidden, cell):
        init = init.unsqueeze(0)
        # init = [1, batch_size, output_dim]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        embedding = self.dropout_fun(self.coordinate2emb(init))
        # print("embedding",embedding.shape)
        output, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        prediction = self.hid2coordinate(output).squeeze(0)
        return prediction, hidden, cell


# target_len 重采样到 max_dst_len
def conditional_decode_seq(
        decoder, output_shape, init, hidden, cell,
        device, teacher_force=0.5, target=None
):
    if teacher_force > 0 and target is None:
        raise NotImplementedError(
            'require stadard sequence as input'
            'when using teacher force'
        )
    target_len, batch_size, output_dim = output_shape
    outputs = torch.zeros(output_shape).to(device)
    current, outputs[0] = init, init
    # print('init',init.shape, init)
    for t in range(1, target_len):

        output, hidden, cell = decoder(current, hidden, cell)
        # print('output',output.shape)
        outputs[t] = output
        current = target[t] if random.random() < teacher_force else output
        # print('current',current.shape)

    return outputs

class ConditionEncoder(torch.nn.Module):
    def __init__(self, branch_encoder, hidden_dim, n_layers=2, dropout=0.5):
        super(ConditionEncoder, self).__init__()
        self.branch_encoder = branch_encoder
        self.path_rnn = torch.nn.LSTM(
            branch_encoder.n_layers * branch_encoder.hidden_dim * 2,
            hidden_dim, n_layers, dropout=dropout
        )
        self.hidden_dim, self.n_layers = hidden_dim, n_layers

    def forward(self, prefix, seq_len, window_len):
        # prefix = [bs, window len, seq_len, data_dim]
        # seq_len = [bs, window len]
        # window_len = [bs]
        bs, wind_l, seq_l, input_dim = prefix.shape
        all_seq_len, all_seq = [], []
        for idx, t in enumerate(window_len):
            all_seq_len.extend(seq_len[idx][:t])
            all_seq.append(prefix[idx][:t])
        all_seq = torch.cat(all_seq, dim=0).permute(1, 0, 2)
        # print('[info] seq_shape', all_seq.shape, sum(window_len))

        h_branch, c_branch = self.branch_encoder(all_seq, all_seq_len)
        # print('[info] hshape', h_branch.shape, c_branch.shape)

        hidden_seq = torch.cat([h_branch, c_branch], dim=0)
        # print('[info] hidden_seq_shape', hidden_seq.shape)
        seq_number = sum(window_len)
        inter_dim = self.branch_encoder.n_layers * \
            self.branch_encoder.hidden_dim * 2
        hidden_seq = hidden_seq.transpose(0, 1).reshape(seq_number, -1)
        all_hidden = torch.zeros((bs, wind_l, inter_dim)).to(hidden_seq)
        curr_pos = 0
        for idx, t in enumerate(window_len):
            all_hidden[idx][:t] = hidden_seq[curr_pos: curr_pos + t]
            curr_pos += t
        assert curr_pos == seq_number, 'hidden vars dispatched error'

        all_hidden = all_hidden.permute(1, 0, 2)
        # print('[info] all hidden shape', all_hidden.shape, len(window_len))
        packed_wind = pack_padded_sequence(
            all_hidden, window_len, enforce_sorted=False
        )
        _, (h_path, c_path) = self.path_rnn(packed_wind)
        return h_path, c_path


class ConditionalSeq2SeqVAE(torch.nn.Module):
    def __init__(self, encoder, decoder, distribution, tgnn, device, forgettable=None, remove_path=False,
                remove_global=False, new_model=False, dropout=0.1):
        super(ConditionalSeq2SeqVAE, self).__init__()
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.distribution = distribution
        self.new_model = new_model
        if new_model:
            self.condition_encoder = ConditionEncoder(self.encoder, self.encoder.hidden_dim, self.encoder.n_layers, dropout=dropout)
        # forgettable == None mean pooling
        # otherwise h_path[w] = forgettable * h_path[w-1]
        #                       + (1 - forgettable) * raw_embedding[w-1]
        self.forgettable = forgettable if forgettable != 0 else None

        print("**************************************")
        print(remove_global, remove_path)
        print("**************************************")

        self.tgnn = tgnn.to(device)
        self.global_dim = self.tgnn.size
        self.remove_global = remove_global
        self.remove_path = remove_path

        mean = torch.full([1,encoder.hidden_dim],0.0)
        std = torch.full([1,encoder.hidden_dim],1.0)
        self.gauss = torch.distributions.Normal(mean, std)

        self.state2latent = torch.nn.Linear(
            encoder.hidden_dim * encoder.n_layers * 6 + self.global_dim,
            distribution.lat_dim
        )
        self.latent2state_l = torch.nn.Linear(
            distribution.lat_dim + encoder.hidden_dim * encoder.n_layers * 2 + self.global_dim,
            decoder.hidden_dim * decoder.n_layers * 2
        )
        self.latent2state_r = torch.nn.Linear(
            distribution.lat_dim + encoder.hidden_dim * encoder.n_layers * 2 + self.global_dim,
            decoder.hidden_dim * decoder.n_layers * 2
        )
        assert encoder.hidden_dim == decoder.hidden_dim, \
            "hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "encoder and decoder must have equal number of layers!"

    def encode(self, src_l, seq_len_l, src_r, seq_len_r, h_path, h_global):
        hidden_l, cell_l = self.encoder(src_l, seq_len_l)
        n_layers, batch_size, hid_dim = hidden_l.shape
        states_l = torch.cat((hidden_l, cell_l), dim=0)
        # result states = [bs, 2*n_layers*hidden_dim]
        states_l = states_l.permute(1, 0, 2).reshape(batch_size, -1)

        hidden_r, cell_r = self.encoder(src_r, seq_len_r)
        states_r = torch.cat((hidden_r, cell_r), dim=0)
        # result states = [bs, 2*n_layers*hidden_dim]
        states_r = states_r.permute(1, 0, 2).reshape(batch_size, -1)

        # 拼接上 h_path
        # 拼接上TGN output h_global，shape = [bs,self.global_dim]
        states = torch.cat((states_l, states_r, h_path, h_global), dim=1)
        h = self.state2latent(states)
        tup, kld, vecs = self.distribution.build_bow_rep(h, n_sample=5)
        Z = torch.mean(vecs, dim=0)
        condition = h_global
        return h, Z, condition

    def _get_decoder_states(self, z, batch_size, decode_left):
        # cat latent with h_path and h_globalbac
        h = z #h为（1,384）
        decoder_states = self.latent2state_r(h).reshape(batch_size, -1, 2)
        hidden_shape = (batch_size, self.decoder.hidden_dim, self.decoder.n_layers)
        hidden = decoder_states[:, :, 0].reshape(*hidden_shape)
        hidden = hidden.permute(2, 0, 1).contiguous()

        cell = decoder_states[:, :, 1].reshape(*hidden_shape)
        cell = cell.permute(2, 0, 1).contiguous()
        return hidden, cell

    def forward(self, noise_branch,smooth_branch,bs,teacher_force=0.5, need_gauss=False):

        output_dim = self.decoder.output_dim

        z = map_to_feature_vector_with_cnn(noise_branch)
        noise_branch = noise_branch.permute(1, 0, 2)
        smooth_branch = smooth_branch.permute(1, 0, 2)

        hidden, cell = self._get_decoder_states(z, bs, False)

        target_len = smooth_branch.shape[0]

        denoise_branch = conditional_decode_seq(
            self.decoder, (target_len, bs, output_dim),
            smooth_branch[0], hidden, cell, self.device,
            teacher_force=teacher_force, target=smooth_branch
        )
        denoise_branch= denoise_branch.permute(1, 0, 2)
        return denoise_branch


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        # 使用Conv1d处理每个点的3个特征，padding=1保持长度不变
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)  # 池化
        self.fc = nn.Linear(64 * 8, 384)  # 假设经过卷积和池化后的维度是64*8

    def forward(self, x):
        x = self.conv1(x)  # Conv1d (batch, 3, 16) -> (batch, 64, 16)
        x = self.pool(x)  # MaxPool1d -> (batch, 64, 8)
        x = x.view(x.size(0), -1)  # 扁平化 (batch, 64*8)
        x = self.fc(x)  # 通过全连接层
        return x


def map_to_feature_vector_with_cnn(noise_branch):
    # 确保输入的noise_branch是Tensor类型，并且位于正确的设备上
    device = noise_branch.device  # 获取输入设备，确保模型和输入在同一设备上

    # 转换输入张量的形状 (bs, 16, 3) -> (bs, 3, 16)
    points_tensor = noise_branch.transpose(1, 2)  # 变为 (bs, 3, 16)
    points_tensor = points_tensor.to(torch.float32)
    model = CNNFeatureExtractor().to(device)  # 将模型移到与输入相同的设备
    feature_vector = model(points_tensor)  # 通过模型处理
    return feature_vector  # 返回形状为 (bs, 384) 的Tensor


class BranchEncRnn(torch.nn.Module):
    def __init__(
        self, input_dim, embedding_dim, hidden_dim,
        n_layers=2, dropout=0.5
    ):
        super(BranchEncRnn, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.coordinate2emb = torch.nn.Linear(input_dim, embedding_dim)
        self.dropout_fun = torch.nn.Dropout(dropout)
        self.rnn = torch.nn.LSTM(
            embedding_dim, hidden_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0
        )

    def forward(self, src, seq_len, return_in_one=False):
        input_seq = self.dropout_fun(self.coordinate2emb(src))
        packed_seq = pack_padded_sequence(
            input_seq, seq_len, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.rnn(packed_seq)

        # outputs = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        #
        if not return_in_one:
            return hidden, cell
        else:
            batch_size = hidden.shape[1]
            answer = torch.cat([hidden, cell], dim=0)
            answer = answer.permute(1, 0, 2).reshape(batch_size, -1)
            return answer


class BranchDecRnn(torch.nn.Module):
    def __init__(
        self, output_dim, embedding_dim, hidden_dim,
        n_layers=2, dropout=0.5
    ):
        super(BranchDecRnn, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.hid2coordinate = torch.nn.Linear(hidden_dim, output_dim)
        self.coordinate2emb = torch.nn.Linear(output_dim, embedding_dim)
        self.dropout_fun = torch.nn.Dropout(dropout)
        self.rnn = torch.nn.LSTM(
            embedding_dim, hidden_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0
        )

    def decode_a_step(self, init, hidden, cell):
        init = init.unsqueeze(0)
        # init = [1, batch_size, output_dim]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        embedding = self.dropout_fun(self.coordinate2emb(init))
        output, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        prediction = self.hid2coordinate(output).squeeze(0)
        return prediction, hidden, cell

    def forward(
        self, hidden, cell, target_len=None, target=None,
        teaching=0.5, init=None
    ):
        if target_len is None and target is None:
            raise ValueError('the target_length should be specified')
        if init is None and target is None:
            raise ValueError('the start point should be specified')
        if teaching > 0 and target is None:
            raise NotImplementedError(
                'require stadard sequence as input'
                'when using teacher force'
            )

        if target_len is None:
            target_len = target.shape[0]
        if init is None:
            init = target[0]

        batch_size = hidden.shape[1]
        output_shape = (target_len, batch_size, self.output_dim)
        outputs = torch.zeros(output_shape).to(hidden.device)
        outputs[0] = init
        current = outputs[0].clone()
        for t in range(1, target_len):
            output, hidden, cell = self.decode_a_step(current, hidden, cell)
            outputs[t] = output
            current = target[t] if random.random() < teaching else output
        return outputs




class RNNAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(RNNAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, src, seq_len, target=None, teaching=0.5,
        init=None, target_len=None
    ):
        hidden, cell = self.encoder(src, seq_len, return_in_one=False)
        return self.decoder(
            hidden, cell, target_len=target_len,
            target=target, init=init, teaching=teaching
        )