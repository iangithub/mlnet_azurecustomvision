using Microsoft.ML;
using mlnettfmodelapp.MLModel;
using System;
using System.Drawing;
using System.IO;
using System.Linq;

namespace mlnettfmodelapp
{
    class Program
    {

        static void Main(string[] args)
        {
            var modelpath = Path.Combine(Environment.CurrentDirectory, "MLModel", "mlmodel.zip");
            var imgspath = Path.Combine(Environment.CurrentDirectory, "ImgDatas");
            var labels = System.IO.File.ReadAllLines(Path.Combine(Environment.CurrentDirectory, "MLModel", "labels.txt"));


            MLContext mlContext = new MLContext(seed: 1);

            // Load the model
            ITransformer loadedModel = mlContext.Model.Load(modelpath, out var schema);

            // Make prediction (input = ImageNetData, output = ImageNetPrediction)
            var predictor = mlContext.Model.CreatePredictionEngine<ImageInputData, ImageLabelPrediction>(loadedModel);

            DirectoryInfo imgdir = new DirectoryInfo(imgspath);
            foreach (var jpgfile in imgdir.GetFiles("*.jpg"))
            {
                var imputimg = new ImageInputData { Image = ConvertToBitmap(jpgfile.FullName) };

                var pred = predictor.Predict(imputimg);

                var imgpredresult = new ImagePredictedLabelWithProbability()
                {
                    Name = jpgfile.Name
                };

                Console.WriteLine($"Filename:{imgpredresult.Name}");

                foreach (var item in pred.PredictedLabels)
                {
                    Console.WriteLine($"Predict Result:{item}");
                }
                var maxresult = pred.PredictedLabels.Max();
                if (maxresult >= 0.7)
                    Console.WriteLine($"Predict Label Result:{labels[pred.PredictedLabels.AsSpan().IndexOf(maxresult)]}");

                Console.WriteLine("===== next image ======");
            }
        }

      
        public static Bitmap ConvertToBitmap(string fileName)
        {
            Bitmap bitmap;
            using (Stream bmpStream = System.IO.File.Open(fileName, System.IO.FileMode.Open))
            {
                Image image = Image.FromStream(bmpStream);

                bitmap = new Bitmap(image);

            }
            return bitmap;
        }
    }
}
