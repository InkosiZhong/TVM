import PyNvCodec as nvc

class ColorConverter:
    """
    Colorspace conversion chain.
    """

    def __init__(self, width: int, height: int, gpu_id: int):
        self.gpu_id = gpu_id
        self.w = width
        self.h = height
        self.chain = []
        self.dst_fmt = None

    def add(self, src_fmt: nvc.PixelFormat, dst_fmt: nvc.PixelFormat) -> None:
        self.chain.append(
            nvc.PySurfaceConverter(self.w, self.h, src_fmt, dst_fmt, self.gpu_id)
        )
        self.dst_fmt = dst_fmt

    def run(self, src_surface: nvc.Surface) -> nvc.Surface:
        surf = src_surface
        cc = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)

        for cvt in self.chain:
            surf = cvt.Execute(surf, cc)
            if surf.Empty():
                raise RuntimeError("Failed to perform color conversion")

        return surf.Clone(self.gpu_id)