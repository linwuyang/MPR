import torch
from torch.nn import functional as F
import torch.nn as nn
import copy

from helper.replayer import Replayer
from helper.continual_learner import ContinualLearner
from helper.utils import l2_loss

def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%Ss"' % noise_type)


class Predictor(ContinualLearner, Replayer):
    '''Model for predicting trajectory, "enriched" as "ContinualLearner"-, Replayer- and ExemplarHandler-object.'''

    # reference GAN code, generator part, encoder & decoder (LSTM)
    def __init__(
            self,
            obs_len,
            pred_len,
            traj_lstm_input_size,
            traj_lstm_hidden_size,
            traj_lstm_output_size,
            dropout=0,
            noise_dim=(8,),
            noise_type="gaussian",
    ):
        super().__init__()
        self.label = "lstm"
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.traj_lstm_input_size = traj_lstm_input_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.traj_lstm_output_size = traj_lstm_output_size

        self.noise_dim = noise_dim
        self.noise_type = noise_type


        #--------------------------MAIN SPECIFY MODEL------------------------#

        #-------Encoder-------#
        self.traj_lstm_model = nn.LSTMCell(traj_lstm_input_size, traj_lstm_hidden_size)

        #-------Decoder------#
        self.pred_lstm_model = nn.LSTMCell(traj_lstm_input_size, traj_lstm_output_size)
        self.pred_hidden2pos =nn.Linear(self.traj_lstm_output_size, 2)

        # for param in self.parameters():
        #     param.requires_grad_(True)

    # initial encoder traj lstm hidden states
    def init_encoder_traj_lstm(self, batch):
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )
    # initial decoder traj lstm hidden states
    def init_decoder_traj_lstm(self, batch):
        return (
            torch.randn(batch, self.traj_lstm_output_size).cuda(),
            torch.randn(batch, self.traj_lstm_output_size).cuda(),
        )

    # add noise before decoder
    def add_noise(self, _input):
        noise_shape = (_input.size(0),) + self.noise_dim
        z_decoder = get_noise(noise_shape, self.noise_type)
        decoder_h = torch.cat([_input, z_decoder], dim=1)
        return decoder_h


    @property
    def name(self):
        return "{}".format("lstm")

    def forward(self, obs_traj_pos, seq_start_end):
        batch = obs_traj_pos.shape[1] #todo define the batch
        traj_lstm_h_t, traj_lstm_c_t = self.init_encoder_traj_lstm(batch)
        # pred_lstm_h_t, pred_lstm_c_t = self.init_decoder_traj_lstm(batch)
        pred_traj_pos = []
        traj_lstm_hidden_states = []
        pred_lstm_hidden_states = []

        # encoder, calculate the hidden states

        for i, input_t in enumerate(
            obs_traj_pos[: self.obs_len].chunk(
                obs_traj_pos[: self.obs_len].size(0), dim=0
            )
        ):
            input_t = input_t.squeeze(0)
            traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(
                input_t, (traj_lstm_h_t, traj_lstm_c_t)
            )
            traj_lstm_hidden_states += [traj_lstm_h_t]


        output = obs_traj_pos[self.obs_len-1]
        pred_lstm_h_t_before_noise = traj_lstm_hidden_states[-1]
        # pred_lstm_h_t = self.add_noise(pred_lstm_h_t_before_noise)
        pred_lstm_h_t = pred_lstm_h_t_before_noise
        pred_lstm_c_t = torch.zeros_like(pred_lstm_h_t).cuda()

        for i in range(self.pred_len):
            
            pred_lstm_h_t, pred_lstm_c_t = self.pred_lstm_model(
                output, (pred_lstm_h_t, pred_lstm_c_t)
            )
            output = self.pred_hidden2pos(pred_lstm_h_t)
            pred_traj_pos += [output]
        
        outputs = torch.stack(pred_traj_pos)

        return outputs



