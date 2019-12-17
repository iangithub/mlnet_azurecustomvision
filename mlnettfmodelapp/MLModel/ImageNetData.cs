using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;

namespace mlnettfmodelapp.MLModel
{
    public class ImageInputData
    {
        [ImageType(224, 224)]
        public Bitmap Image { get; set; }
    }

    public class ImageLabelPrediction
    {
        [ColumnName("loss")]
        public float[] PredictedLabels;
    }

    public class ImagePredictedLabelWithProbability
    {
        public string Name;
        public string PredictedLabel;
        public float Probability { get; set; }
    }
}
