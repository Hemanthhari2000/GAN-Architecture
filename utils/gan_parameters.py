import torch


class GanParameters:
    def __init__(self, dataroot='dataset', ngpu=torch.cuda.device_count(), workers=2, batch_size=128, image_size=128, nc=3, nz=100, ngf=64, ndf=64, num_epochs=1, lr=0.0002, beta1=0.5) -> None:
        self._dataroot = dataroot
        self._ngpu = ngpu
        self._workers = 4 * self._ngpu if self._ngpu > 0 else workers
        self._batch_size = batch_size
        self._image_size = image_size
        self._nc = nc
        self._nz = nz
        self._ngf = ngf
        self._ndf = ndf
        self._num_epochs = num_epochs
        self._lr = lr
        self._beta1 = beta1

    def get_dataroot(self):
        return self._dataroot

    def get_ngpu(self):
        return self._ngpu

    def get_workers(self):
        return self._workers

    def get_batch_size(self):
        return self._batch_size

    def get_image_size(self):
        return self._image_size

    def get_nc(self):
        return self._nc

    def get_nz(self):
        return self._nz

    def get_ngf(self):
        return self._ngf

    def get_ndf(self):
        return self._ndf

    def get_num_epochs(self):
        return self._num_epochs

    def get_lr(self):
        return self._lr

    def get_beta1(self):
        return self._beta1

    def set_dataroot(self, dataroot):
        self._dataroot = dataroot

    def set_ngpu(self, ngpu):
        self._ngpu = ngpu

    def set_workers(self, workers):
        self._workers = workers

    def set_image_size(self, image_size):
        self._image_size = image_size

    def set_nc(self, nc):
        self._nc = nc

    def set_nz(self, nz):
        self._nz = nz

    def set_ngf(self, ngf):
        self._ngf = ngf

    def set_ndf(self, ndf):
        self._ndf = ndf

    def set_num_epochs(self, num_epochs):
        self._num_epochs = num_epochs

    def set_lr(self, lr):
        self._lr = lr

    def set_beta1(self, beta1):
        self._beta1 = beta1


# ganParameters = GanParameters()

# print(ganParameters.get_dataroot())
# print(ganParameters.set_dataroot('Hello'))
# print(ganParameters.get_dataroot())
