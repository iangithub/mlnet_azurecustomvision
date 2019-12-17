using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace mlnettfmodel.MLModel
{
    public class ModelCreator
    {
        private MLContext _mlcontext;
        private ITransformer _mlmodel;

        public ModelCreator()
        {
            _mlcontext = new MLContext(seed: 1);
        }

        public void TrainAndSaveModel()
        {
            //由Azure Custom Vision訓練後導出的model (TensorFlow Model)
            var tfmodelpath = Path.Combine(Environment.CurrentDirectory,"TFModel", "model.pb");
            //由ML.NET轉換後，要儲存的model
            var savemodelpath = Path.Combine(Environment.CurrentDirectory,"MLModel", "mlmodel.zip");

            _mlmodel = TrainModel(tfmodelpath);
            _mlcontext.Model.Save(_mlmodel, null, savemodelpath);
        }

        /// <summary>
        /// Img Settings
        /// 參數值使用 Netron 工具開啟原始TensorFlow Model，可以得知
        /// </summary>
        private struct TrainModelSettings
        {
            public const int ImgHeight = 224;
            public const int ImgWidth = 224;
            public const float Mean = 117;         //offsetImage
            public const bool ChannelsLast = true; //interleavePixelColors
            public const string InputTensorName = "Placeholder";// input tensor name
            public const string OutputTensorName = "loss"; // output tensor name
        }


        private ITransformer TrainModel(string tfmdelpath)
        {
            /*
             Transforms.ResizeImages 
             Transforms.ExtractPixels 將圖片轉換為含像素的矩陣
             LoadTensorFlowModel 載入TensorFlow模型
             */
            var pipeline = _mlcontext.Transforms.ResizeImages(outputColumnName: TrainModelSettings.InputTensorName
                                                                , imageWidth: TrainModelSettings.ImgWidth
                                                                , imageHeight: TrainModelSettings.ImgHeight
                                                                , inputColumnName: nameof(ImageInputData.Image))
                .Append(_mlcontext.Transforms.ExtractPixels(outputColumnName: TrainModelSettings.InputTensorName
                                                            , interleavePixelColors: TrainModelSettings.ChannelsLast
                                                            , offsetImage: TrainModelSettings.Mean))
                .Append(_mlcontext.Model.LoadTensorFlowModel(tfmdelpath)
                .ScoreTensorFlowModel(outputColumnNames: new[] { TrainModelSettings.OutputTensorName },
                                    inputColumnNames: new[] { TrainModelSettings.InputTensorName },
                                    addBatchDimensionInput: false));

            ITransformer mlModel = pipeline.Fit(CreateEmptyDataView());

            return mlModel;
        }

        /// <summary>
        /// 由於ML.NET 在訓練Model時，須給與訓練集資料
        /// 本例只是將Azure Custom Vision導出的模型
        /// 再轉化為ML.NET 模型，並不需要重新訓練Model
        /// 故直接建立一個假的Image資料即可(Schema要符合)
        /// </summary>
        /// <returns></returns>
        private IDataView CreateEmptyDataView()
        {
            List<ImageInputData> list = new List<ImageInputData>();
            list.Add(new ImageInputData()
            {
                Image = new System.Drawing.Bitmap(TrainModelSettings.ImgWidth , TrainModelSettings.ImgHeight)
            });

            var result = _mlcontext.Data.LoadFromEnumerable<ImageInputData>(list);
            return result;
        }
    }
}
