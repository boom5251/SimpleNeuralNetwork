using System;
using System.IO;



namespace NeuralNetworkConsoleApp.Data
{
    public class CsvParser
    {
        public static double[,] GetDatasetArr(string fileUri)
        {
            double[,] datasetArr;

            StreamReader sr = new StreamReader(fileUri);
            string[] headers = sr.ReadLine().Split(',', ';');
            string data = sr.ReadToEnd();
            sr.Close();

            string[] rowStrs = data.Split('\n');


            datasetArr = new double[rowStrs.Length, headers.Length];

            for (int row = 0; row < rowStrs.Length; row++)
            {
                string[] currentRow = rowStrs[row].Split(';', ',');
                for (int column = 0; column < headers.Length; column++)
                {
                    datasetArr[row, column] = Convert.ToDouble(currentRow[column].Replace('.', ','));
                    Console.Write(datasetArr[row, column] + "; ");
                }
                Console.WriteLine();
            }

            return datasetArr;
        }
    }
}
