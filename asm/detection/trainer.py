import torch


class trainer:
    def __init__(self):
        self.training_status = {
            'status': "idle",
            'stage': "",
            'epoch_trained': 0,
            'epoch_current': 0,
            'epoch_total': 10,
            'tb_path': '',
            # all custom evaluations should be put in msgs, formatting like msgs{"loss": 0.234,...}
            'msgs': {},
        }

        self.need_stop = False
        self.model = None

    def terminate(self):
        self.need_stop = True
        self.training_status['status'] = 'idle'
        self.training_status['stage'] = 'stopping'

    def clear_training_status(self):
        self.training_status['loss'] = 0
        self.training_status['train_map'] = 0
        self.training_status['test_map'] = 0
        self.training_status['valid_map'] = 0
        self.training_status['train_recall'] = 0
        self.training_status['valid_recall'] = 0
        self.training_status['test_recall'] = 0
        self.training_status['msgs'] = {}
        self.need_stop = False

    def load_dataset(self):
        raise NotImplementedError('please implement this function')

    def train(self):
        raise NotImplementedError('please implement this function')

    def save_weights(self, path):
        if self.model:
            torch.save(self.model.state_dict(), path)

    def update_status(self, status_dict={}):
        if self.update_fn is not None:
            self.training_status.update(status_dict)
            self.update_fn(self.training_status)
        return self.need_stop


if __name__ == '__main__':
    pass
