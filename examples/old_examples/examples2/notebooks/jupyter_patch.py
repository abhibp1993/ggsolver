# This part is for saving images while exporting to HTML
import base64, io, IPython
from PIL import Image as PILImage
from IPython.display import Image


def img2html(fpath):
    image = PILImage.open(fpath)
    output = io.BytesIO()
    image.save(output, format='PNG')
    encoded_string = base64.b64encode(output.getvalue()).decode()
    return '<img src="data:image/png;base64,{}"/>'.format(encoded_string)