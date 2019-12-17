using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;

namespace mlnettfmodel.MLModel
{
    public class ImageInputData
    {
        [ImageType(224, 224)]
        public Bitmap Image { get; set; }
    }
}
