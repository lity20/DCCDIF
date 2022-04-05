from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F


class CodeCloud(nn.Module):
    def __init__(self, config, num_records):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Building CodeCloud.')
        super().__init__()
        self.config = config

        self.codes_position = nn.Parameter(torch.rand(num_records, config['num_codes'], 3) - 0.5)
        self.codes = nn.Parameter(torch.randn(num_records, config['num_codes'], config['code_dim']) * 0.01)

        num_params = sum(p.data.nelement() for p in self.parameters())
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'CodeCloud done(#parameters=%d).'%num_params)

    def query(self, indices, query_points):
        """
        Args:
            indices: tensor, (batch_size,)
            query_points: tensor, (batch_size, num_points, 3)
        Returns:
            query_codes: tensor, (batch_size, num_points, code_dim)
            square_dist: tensor, (batch_size, num_points, num_codes)
            weight: tensor, (batch_size, num_points, num_codes)
        """
        batch_codes_position = self.codes_position[indices]
        batch_codes = self.codes[indices]
        square_dist = (query_points.unsqueeze(2) - batch_codes_position.unsqueeze(1)).pow(2).sum(dim=-1) + 1e-16
        weight = 1.0 / (torch.sqrt(square_dist) ** self.config['dist_scale'])
        weight = weight / weight.sum(dim=-1, keepdim=True)
        query_codes = torch.matmul(weight, batch_codes)
        return query_codes, square_dist, weight


class IM_Decoder(nn.Module):
    def __init__(self, input_dim):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Building IM-decoder.')
        super().__init__()

        self.linear_1 = nn.Linear(input_dim, 2048, bias=True)
        self.linear_2 = nn.Linear(input_dim+2048, 1024, bias=True)
        self.linear_3 = nn.Linear(input_dim+1024, 512, bias=True)
        self.linear_4 = nn.Linear(input_dim+512, 256, bias=True)
        self.linear_5 = nn.Linear(input_dim+256, 128, bias=True)
        self.linear_6 = nn.Linear(128, 1, bias=True)

        num_params = sum(p.data.nelement() for p in self.parameters())
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'IM decoder done(#parameters=%d).'%num_params)

    def forward(self, batch_input):
        l1 = self.linear_1(batch_input)
        l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)
        l1 = torch.cat([l1, batch_input], dim=-1)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)
        l2 = torch.cat([l2, batch_input], dim=-1)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)
        l3 = torch.cat([l3, batch_input], dim=-1)

        l4 = self.linear_4(l3)
        l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)
        l4 = torch.cat([l4, batch_input], dim=-1)

        l5 = self.linear_5(l4)
        l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

        l6 = self.linear_6(l5)
        return l6.squeeze(-1)


class Network(nn.Module):
    def __init__(self, config, num_records):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Building network.')
        super().__init__()
        self.config = config

        self.code_cloud = CodeCloud(config['code_cloud'], num_records)
        self.decoder = IM_Decoder(config['code_cloud']['code_dim']+3)

        num_params = sum(p.data.nelement() for p in self.parameters())
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Network done(#parameters=%d).'%num_params)
    
    def forward(self, indices, query_points):
        """
        Args:
            indices: tensor, (batch_size,)
            query_points: tensor, (batch_size, num_points, 3)
        Returns:
            pred_sd: tensor, (batch_size, num_points)
        """
        query_codes, square_dist, weight = self.code_cloud.query(indices, query_points)
        batch_input = torch.cat([query_codes, query_points], dim=-1)
        pred_sd = self.decoder(batch_input)

        self.per_step_query_codes = query_codes
        self.per_step_square_dist = square_dist
        self.per_step_weight = weight
        self.per_step_pred_sd = pred_sd
        return pred_sd

    def loss(self, gt_sd):
        loss_dict = {}
        loss_dict['total_loss'] = 0

        l2_loss = F.mse_loss(self.per_step_pred_sd, gt_sd)
        loss_dict['l2_loss'] = l2_loss
        loss_dict['total_loss'] += l2_loss

        if self.config['code_regularization_lambda'] > 0:
            code_regularization_loss = torch.mean(torch.linalg.norm(self.per_step_query_codes, dim=-1))
            loss_dict['code_regularization_loss'] = code_regularization_loss
            loss_dict['total_loss'] += code_regularization_loss * self.config['code_regularization_lambda']

        if self.config['code_position_lambda'] > 0:
            dist = self.per_step_square_dist * self.per_step_weight.detach()
            with torch.no_grad():
                error = torch.abs(self.per_step_pred_sd - gt_sd).unsqueeze(-1)
            code_position_loss = torch.mean(dist * error)
            loss_dict['code_position_loss'] = code_position_loss
            loss_dict['total_loss'] += code_position_loss * self.config['code_position_lambda']

        return loss_dict
