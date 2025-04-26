import torch.nn as nn
import torch.nn.functional as F
import torch

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, tgram, decode_function=F.log_softmax):
        super(Seq2Seq, self).__init__()
        self.tgram = tgram
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

    def flatten_parameters(self):
        pass

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0, waves = None):
        if self.tgram != None:
            tgram_output = self.tgram(waves)
            tgram_output = tgram_output.unsqueeze(1)
            input_variable = torch.cat([input_variable, tgram_output], 2)

        self.encoder.rnn.flatten_parameters()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)

        self.decoder.rnn.flatten_parameters()
        decoder_output = self.decoder(inputs=target_variable,
                                      encoder_hidden=None,
                                      encoder_outputs=encoder_outputs,
                                      function=self.decode_function,
                                      teacher_forcing_ratio=teacher_forcing_ratio)

                                                                
        return decoder_output

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params