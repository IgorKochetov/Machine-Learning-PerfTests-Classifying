using System;
using System.Text;

namespace NetFest_NN
{
    public class Classification
    {
        public Classification(double[] classifications)
        {
            Classifications = classifications;
        }

        public double[] Classifications { get; }

        public override string ToString()
        {
            var sb = new StringBuilder();
            for (int i = 0; i < Classifications.Length; i++)
            {
                sb.Append($"{i}: {Classifications[i]:p0}{Environment.NewLine}");
            }
            return sb.ToString();
        }
    }
}
