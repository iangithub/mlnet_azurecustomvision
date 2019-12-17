using mlnettfmodel.MLModel;
using System;

namespace mlnettfmodel
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlcreator = new ModelCreator();
            mlcreator.TrainAndSaveModel();
            Console.WriteLine("Hello ML NET!");
        }
    }
}
