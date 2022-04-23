import enum
import torch
import torch_scatter

class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9

def triangles_to_edges(faces):
    edges = torch.cat((faces[:, :, 0:2], faces[:, :, 1:3], torch.stack((faces[:, :, 2], faces[:, :, 0]), dim=2)), dim=1)
    receivers, _ = torch.min(edges, dim=2)
    senders, _ = torch.max(edges, dim=2)
    packed_edges = torch.stack((senders, receivers), dim=2)
    unique_edges = torch.unique(packed_edges, return_inverse=False, return_counts=False, dim=1)
    senders, receivers = torch.unbind(unique_edges, dim=2)
    senders = senders.to(torch.int64)
    receivers = receivers.to(torch.int64)
    two_way_connectivity = (torch.cat((senders, receivers), dim=1), torch.cat((receivers, senders), dim=1))
    return {'two_way_connectivity': two_way_connectivity, 'senders': senders, 'receivers': receivers}

class Normalizer(torch.nn.Module):
    """Feature normalizer that accumulates statistics online."""

    def __init__(self, device, size, batchsize, max_accumulations=10 ** 6, std_epsilon=1e-8):
        super(Normalizer, self).__init__()
     
        self.device = device
        self.batchsize = batchsize
        self._max_accumulations = max_accumulations
        self._std_epsilon = torch.tensor([std_epsilon], requires_grad=False).to(self.device)

        self._acc_count = torch.zeros(1, dtype=torch.float32, requires_grad=False).to(self.device)
        self._num_accumulations = torch.zeros(1, dtype=torch.float32, requires_grad=False).to(self.device)
        self._acc_sum = torch.zeros((self.batchsize, size), dtype=torch.float32, requires_grad=False).to(self.device)
        self._acc_sum_squared = torch.zeros((self.batchsize, size), dtype=torch.float32, requires_grad=False).to(self.device)

    def forward(self, batched_data, node_num=None, accumulate=True):
        """Normalizes input data and accumulates statistics."""
        if accumulate and self._num_accumulations < self._max_accumulations:
            # stop accumulating after a million updates, to prevent accuracy issues
            self._accumulate(batched_data)
        return (batched_data - torch.unsqueeze(self._mean(), dim=1)) / torch.unsqueeze(self._std_with_epsilon(), dim=1)

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data, node_num=None):
        """Function to perform the accumulation of the batch_data statistics."""
        count = torch.tensor(batched_data.shape[1], dtype=torch.float32, device=self.device)

        data_sum = torch.sum(batched_data, dim=1)
        squared_data_sum = torch.sum(batched_data ** 2, dim=1)
        self._acc_sum = self._acc_sum.add(data_sum)
        self._acc_sum_squared = self._acc_sum_squared.add(squared_data_sum)
        self._acc_count = self._acc_count.add(count)
        self._num_accumulations = self._num_accumulations.add(1.)

    def _mean(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor([1.], device=self.device))
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor([1.], device=self.device))
        std = torch.sqrt(self._acc_sum_squared / safe_count - self._mean() ** 2)
        return torch.maximum(std, self._std_epsilon)

    def get_acc_sum(self):
        return self._acc_sum
        
def unsorted_segment_operation(device, data, segment_ids, num_segments, operation):
        """
        Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

        :param data: A tensor whose segments are to be summed.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """
        assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

        # segment_ids is a 1-D tensor repeat it to have the same shape as data
        data = data.to(device)
        segment_ids = segment_ids.to(device)
        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long().to(device)
            segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:]).to(device)

        assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

        shape = [num_segments] + list(data.shape[1:])
        result = torch.zeros(*shape).to(device)
        if operation == 'sum':
            result = torch_scatter.scatter_add(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'max':
            result, _ = torch_scatter.scatter_max(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'mean':
            result = torch_scatter.scatter_mean(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'min':
            result, _ = torch_scatter.scatter_min(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'std':
            result = torch_scatter.scatter_std(data.float(), segment_ids, out=result, dim=0, dim_size=num_segments)
        else:
            raise Exception('Invalid operation type!')
        result = result.type(data.dtype)
        return result