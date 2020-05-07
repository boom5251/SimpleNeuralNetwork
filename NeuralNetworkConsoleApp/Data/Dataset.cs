using System;



namespace NeuralNetworkConsoleApp.Data
{
    public class Dataset
    {
        public Dataset(double[,] datasetArr, Type[] columsTypes, int resultsColumnIndex)
        {
            if (datasetArr.GetLength(1) == columsTypes.Length)
            {
                ColumnsTypes = columsTypes;
                ResultColumnIndex = resultsColumnIndex;

                RowsCount = datasetArr.GetLength(0);
                ColumnsCount = datasetArr.GetLength(1);

                Data = new double[RowsCount, ColumnsCount];
                for (int i = 0; i < RowsCount; i++)
                {
                    for (int k = 0; k < ColumnsCount; k++)
                    {
                        Data[i, k] = datasetArr[i, k];
                    }
                }
            }
            else
            {
                throw new IndexOutOfRangeException("Количество элементов массива columsTypes не соответсвует количесву столбцов двумерного массива datasetArr!");
            }
        }



        public Type[] ColumnsTypes { get; }
        public double[,] Data { get; }
        public int ResultColumnIndex { get; }
        public int RowsCount { get; }
        public int ColumnsCount { get; }
        


        public Type GetColumnType(int columnIndex, out double[] column)
        {
            int length = Data.GetLength(0);
            column = new double[length];

            for (int i = 0; i < length; i++)
            {
                column[i] = Data[i, columnIndex];
            }

            return ColumnsTypes[columnIndex];
        }



        public double[] GetRow(int rowIndex, bool withoutResult = false)
        {
            int length = Data.GetLength(1);
            double[] row;

            if (!withoutResult)
            {
                row = new double[length];

                for (int k = 0; k < length; k++)
                {
                    row[k] = Data[rowIndex, k];
                }
            }
            else
            {
                row = new double[length - 1];

                int j = 0;
                for (int k = 0; k < length; k++)
                {
                    if (k != ResultColumnIndex)
                    {
                        row[j] = Data[rowIndex, k];
                        j++;
                    }
                }
            }

            return row;
        }
    }
}
