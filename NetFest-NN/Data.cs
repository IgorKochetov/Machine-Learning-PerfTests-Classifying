using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NetFest_NN
{
	public enum ClassificationColumn
	{
		None,
		First,
		Last,
	}

	public class Data
	{
		public int InputCount => Inputs.First().Length;

		public int ClassCount => Outputs.First().Length;

		public int SampleCount => Inputs.Length;

		public double[][] Inputs { get; set; }

		public double[][] Outputs { get; set; }

		public void Normalize()
		{
			for (int i = 0; i < Inputs.Length; i++)
			{
				double max = Inputs[i].Max();
				for (int j = 0; j < Inputs[i].Length; j++)
				{
					Inputs[i][j] = (Inputs[i][j] / max) * 0.98 + 0.01;
				}
			}
		}

		public void Normalize(double max)
		{
			for (int i = 0; i < Inputs.Length; i++)
			{
				for (int j = 0; j < Inputs[i].Length; j++)
				{
					Inputs[i][j] = (Inputs[i][j] / max) * 0.98 + 0.01;
				}
			}
		}

		public static Data Load(string filepath, ClassificationColumn classificationColumn = ClassificationColumn.None)
		{
			var rows = LoadInternal(filepath).ToArray();
			if (!rows.Any())
			{
				throw new InvalidDataException("No samples found in file");
			}

			var columnCount = rows.First().Length;
			if (rows.Any(x => x.Length != columnCount))
			{
				throw new InvalidDataException("Mixed column lengths");
			}

			var data = new Data
			{
				Inputs = new double[rows.Length][],
				Outputs = new double[rows.Length][],
			};

			var classifications = DetermineClassifications(classificationColumn, rows);

			for (int i = 0; i < rows.Length; i++)
			{
				data.Inputs[i] = new double[classificationColumn != ClassificationColumn.None ? columnCount -1 : columnCount];
				data.Outputs[i] = new double[classifications];

				for (int j = 0; j < columnCount; j++)
				{
					if (classificationColumn == ClassificationColumn.First)
					{
						if (j == 0)
						{
							int classificationIndex = (int)rows[i][j]; // TODO: Better bucket control of classifications
							data.Outputs[i][classificationIndex] = 0.99;
						}
						else
						{
							data.Inputs[i][j-1] = rows[i][j];
						}
					}
					else if (classificationColumn == ClassificationColumn.Last && j + 1 == columnCount)
					{
						int classificationIndex = (int)rows[i][j]; // TODO: Better bucket control of classifications
						data.Outputs[i][classificationIndex] = 0.99;
					}
					else
					{
						data.Inputs[i][j] = rows[i][j];
					}
				}
			}

			return data;
		}

		private static int DetermineClassifications(ClassificationColumn classificationColumn, double[][] rows)
		{
			int classifications = 0;
			if (classificationColumn == ClassificationColumn.First)
			{
				classifications = (int)rows.Select(x => x.First()).Max() + 1;
			}
			else if (classificationColumn == ClassificationColumn.Last)
			{
				classifications = (int)rows.Select(x => x.Last()).Max() + 1;
			}
			return classifications;
		}

		private static IEnumerable<double[]> LoadInternal(string filepath)
		{
			using (var reader = File.OpenText(filepath))
			{
				for (var line = reader.ReadLine(); line != null; line = reader.ReadLine())
				{
					var columns = line.Split(';', ',');
					yield return columns
						.Where(x => !x.Any(char.IsLetter)) // Ignore columns with text in them
						.Select(double.Parse)
						.ToArray();
				}
			}
		}

		public Data Take(double amount = 0.25)
		{
			if (amount <= 0 || amount >= 1.0)
			{
				throw new ArgumentOutOfRangeException(nameof(amount), "Amount must be between ]0;1[");
			}

			var rnd = new Random((int)(DateTime.UtcNow.Ticks % int.MaxValue));

			int rowsToTake = (int)(SampleCount * amount);
			if (rowsToTake < 1)
			{
				throw new InvalidDataException("Not enough rows");
			}

			var data = new Data();
			data.Inputs = new double[rowsToTake][];
			data.Outputs = new double[rowsToTake][];

			for (int i = 0; i < rowsToTake; i++)
			{
				int take = rnd.Next(0, SampleCount);
				data.Inputs[i] = Inputs[take];
				data.Outputs[i] = Outputs[take];
				Inputs = Inputs.RemoveAt(take);
				Outputs = Outputs.RemoveAt(take);
			}

			return data;
		}
	}
}
