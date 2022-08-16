using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization.Json;
using System.Runtime.Serialization.Formatters;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;



namespace MoldableNetwork
{
    
    internal class Neuron
    {
        Random random = new Random();
        Dictionary<Neuron, Func<float, float>> connection;
        float w, b;
        Func<float, float> wFunc;
        Func<float, float> bFunc;
        public Neuron() {
            w = 2f*(0.5f-random.NextSingle());
            b = 2f * (0.5f - random.NextSingle());
        }
        float moldedWActivation(float inp)
        {
            float ret = w * inp;
            // w = ;  // simulate a physical trace
            return ret;
        }
        float moldedBActivation()
        {
            float ret = b;
            return ret;
        }
        float ReLU6(float x)
        {
            return MathF.Max(MathF.Min(6f, x), 0f);
        }
        float moldedActivation(float x)
        {
            float output = w * x + b;
            float diff = output - x;
            return ReLU6(output); // moldedWActivation(x) + moldedBActivation());
        }
    }
    class Network
    {
        public BasicNet bn;
        public Network()
        {
            int nInputs = 3;
            // input maybe x y away from the guy
            bn = new BasicNet(nInputs, 2, 3, 3); // fiddle with number of layers and height of layers
            float[][] x = MatrixMaths.Init2DMatrix(nInputs, 1);
            /*x[0][0] = 20;
            x[1][0] = 10;*/
            // the weights have been initialized with a normal distribution N(0,1)
            // this is INPLACE!!! perturbation bn.PerturbateNetwork(0.01);
            /*
            MatrixMaths.PrintMatrix(bn.WIH);
            MatrixMaths.PrintMatrix(bn.BI);
            MatrixMaths.PrintMatrix(bn.forward(x));
            */

            //for (int i = 0; i < 1000; i++)
            // {
            // output of the environment is the input into the agent, (x,y), (d,phi)
            /// float[][] x = MatrixMaths.Init2DMatrix(nInputs, 1);
            /// x = env.step(output_of_agent); // returns what he sees !!! replace
            // x is the input into the agent
            /// output_of_agent = bn.forward(x);
            // output_of_agent, the environment or the custom program updates its state
            //     to be able to give the input to the agent
            //}

            // here you choose the best agents and copy them into others
            // then restart
            // also: when doing the perturbations it's good to try two or more levels of noise, e.g. 0.01 and 0.05
            //      the better the agents get, smaller perturbations might be more effective

            /// outline of training:
            // 1. generate:
            //          d1 d2 d3 d4 d5 d6
            // 2. play a session
            // 3. choose best guys:
            //          best are d4 and d5
            // 4. copy best guys into others:
            //          d1.copyWeightsFrom(d4), d2.copyWeightsFrom(d4), d3.copyWeightsFrom(d5), d6.copyWeightsFrom(d5)
            // 5. perturb the weights of each guy d1-d6
            //     usually the perturbation magnitude is 0.01-0.1
            // 6. go to 2.
            // 7. when finished, create a way to save and load the weights !!
        }

        public class MatrixMaths
        {
            static Random rand = new Random(); //reuse this if you are generating many
            public static float[][] Transpose(float[][] A)
            {
                float[][] ret = new float[A[0].Length][];
                for(int i = 0; i < A.Length; i++)
                {
                    ret[i] = new float[A.Length];
                    for(int j = 0; j < A[i].Length; j++)
                    {
                        ret[i][j] = A[j][i];
                    }
                }
                return ret;
            }
            public static float[][] MultiplyMatrix(float[][] A, float[][] B)
            {
                int rA = A.GetLength(0);
                int cA = A[0].GetLength(0);
                int rB = B.GetLength(0);
                int cB = B[0].GetLength(0);
                float temp = 0;
                float[][] ret = new float[rA][];
                for (int i = 0; i < rA; i++)
                {
                    ret[i] = new float[cB];
                }
                if (cA != rB)
                {
                    throw new Exception("Column != row number: columns: " + cA + " rows: " + rB);
                }
                else {
                    for (int i = 0; i < rA; i++)
                    {
                        for (int j = 0; j < cB; j++)
                        {
                            temp = 0;
                            for (int k = 0; k < cA; k++)
                            {
                                temp += A[i][k] * B[k][j];
                            }
                            ret[i][j] = temp;
                        }
                    }
                    return ret;
                }
            }
            public static void PrintMatrix(float[][] mat)
            {

                for (int i = 0; i < mat.GetLength(0); i++)
                {
                    for (int j = 0; j < mat[0].GetLength(0); j++)
                    {
                        Console.Write(mat[i][j].ToString("0.00") + " ");
                    }
                    Console.WriteLine();
                }
            }
            public static float NormalNumber()
            {
                float u1 = 1.0f - rand.NextSingle(); //uniform(0,1] random floats
                float u2 = 1.0f - rand.NextSingle();
                return MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Sin(2.0f * MathF.PI * u2); //random normal(0,1)
            }
            public static void InitializeNormalMatrix(ref float[][] mat)
            {
                for (int i = 0; i < mat.GetLength(0); i++)
                {
                    for (int j = 0; j < mat[0].GetLength(0); j++)
                    {
                        mat[i][j] = NormalNumber();
                    }
                }
            }
            public static void InitializeNormalMatrices(ref float[][][] mats)
            {
                for (int i = 0; i < mats.GetLength(0); i++)
                {
                    InitializeNormalMatrix(ref mats[i]);
                }
            }
            public static float[][] Init2DMatrix(int height, int width)
            {
                float[][] mat = new float[height][];
                for (int i = 0; i < height; i++)
                {
                    mat[i] = new float[width];
                }
                return mat;
            }
            public static float[][][] Init3DMatrix(int length, int height, int width)
            {
                float[][][] mat = new float[length][][];
                for (int i = 0; i < length; i++)
                {
                    mat[i] = Init2DMatrix(height, width);
                }
                return mat;
            }
            public static float[][] AddMatrices(float[][] vec1, float[][] vec2)
            {
                float[][] ret = Init2DMatrix(vec1.GetLength(0), vec1[0].GetLength(0));
                for (int i = 0; i < vec1.GetLength(0); i++)
                {
                    for (int j = 0; j < vec1[0].GetLength(0); j++)
                    {
                        ret[i][j] = vec1[i][j] + vec2[i][j];
                    }
                }
                return ret;
            }
            // A - B
            public static float[] SubtractMatrices(float[] A, float[] B)
            {
                float[] ret = new float[A.Length];
                for (int i = 0; i < A.Length; i++)
                {
                    ret[i] = A[i] - B[i];
                }
                return ret;
            }
            // A - B
            public static float[][] SubtractMatrices(float[][] A, float[][] B)
            {
                float[][] ret = Init2DMatrix(A.GetLength(0), A[0].GetLength(0));
                for (int i = 0; i < A.GetLength(0); i++)
                {
                    for (int j = 0; j < A[0].GetLength(0); j++)
                    {
                        ret[i][j] = A[i][j] - B[i][j];
                    }
                }
                return ret;
            }
            public static float[][][] AddMatrices3D(float[][][] vec1, float[][][] vec2)
            {
                float[][][] ret = Init3DMatrix(vec1.GetLength(0), vec1[0].GetLength(0), vec1[0][0].GetLength(0));
                for (int i = 0; i < vec1.GetLength(0); i++)
                {
                    ret[i] = AddMatrices(vec1[i], vec2[i]); // adding two 2D matrices that are an element in the list of 2D matrices
                }
                return ret;
            }
            public static float[] MultiplyVectorByScalar(float[] x, float scalar)
            {
                float[] ret = new float[x.Length];
                for (int i = 0; i < x.Length; i++)
                {
                    ret[i] = x[i] * scalar;
                }
                return ret;
            }
            public static float[][] MultiplyMatrixByScalar(float[][] x, float scalar)
            {
                float[][] ret = new float[x.Length][];
                for (int i = 0; i < x.Length; i++)
                {
                    ret[i] = MultiplyVectorByScalar(x[i], scalar);
                }
                return ret;
            }
            public static float[][][] MultiplyMatricesByScalar(float[][][] x, float scalar)
            {
                float[][][] ret = new float[x.Length][][];
                for (int i = 0; i < x.Length; i++)
                {
                    ret[i] = MultiplyMatrixByScalar(x[i], scalar);
                }
                return ret;
            }
        }

        public static class Losses
        {
            public static float L2Loss(float output, float label)
            {
                float diff = label-output;
                return 0.5f*diff*diff;
            }
            public static float L2Loss(float[] output, float[] label)
            {
                float ret = 0f;
                for (int i = 0; i < output.Length; ++i)
                {
                    ret += L2Loss(output, label);
                }
                return ret;
            }
            public static float L2Loss(float[][] output, float[][] label)
            {
                float ret = 0f;
                for (int i = 0; i < output.Length; ++i)
                {
                    float diff = L2Loss(output, label);
                    ret += diff * diff;
                }
                return ret;
            }
            public static float[] Softmax(float[] vector)
            {
                float expSum = 0f;
                for(int i = 0;i < vector.Length; ++i)
                {
                    expSum += MathF.Exp(vector[i]);
                }
                float invExpSum = 1f / expSum;
                float[] ret = new float[vector.Length];
                for(int i = 0;i < ret.Length; ++i)
                {
                    ret[i] = vector[i] * invExpSum;
                }
                return ret;
            }
            public static float[][] Softmax(float[][] vectors)
            {
                float expSum = 0f;
                for (int i = 0; i < vectors.Length; ++i)
                {
                    for (int j = 0; j < vectors[i].Length; ++j)
                    {
                        expSum += MathF.Exp(vectors[i][j]);
                    }
                }
                float invExpSum = 1f / expSum;
                float[][] ret = new float[vectors.Length][];
                for (int i = 0; i < ret.Length; ++i)
                {
                    ret[i] = new float[vectors[i].Length];
                    for (int j = 0; j < vectors[i].Length; ++j)
                    {
                        ret[i][j] = vectors[i][j] * invExpSum;
                    }
                }
                return ret;
            }
        }

        public static class Activations
        {
            public static float ReLU6(float x)
            {
                return MathF.Max(MathF.Min(6f, x), 0f);
            }
            public static float[] ReLU6(float[] x)
            {
                float[] ret = new float[x.Length];
                for (int i = 0; i < x.Length; i++)
                {
                    ret[i] = ReLU6(x[i]);
                }
                return ret;
            }
            public static float[][] ReLU6(float[][] x)
            {
                float[][] ret = new float[x.Length][];
                for (int i = 0; i < x.Length; i++)
                {
                    ret[i] = ReLU6(x[i]);
                }
                return ret;
            }
            public static float ReLU(float x) // most popular non-linear function
            {
                // shape: __/
                //         0
                return Math.Max(0, x); // popular because its derivative is 1(x) = 0 if x < 0 else 1
            }
            public static float[] ReLU(float[] x)
            {
                float[] ret = new float[x.Length];
                for (int i = 0; i < x.Length; i++)
                {
                    ret[i] = ReLU(x[i]);
                }
                return ret;
            }
            public static float[][] ReLU(float[][] x)
            {
                float[][] ret = new float[x.Length][];
                for (int i = 0; i < x.Length; i++)
                {
                    ret[i] = ReLU(x[i]);
                }
                return ret;
            }
        }
        public static class Backprops
        {
            // TODO: maybe try SGD, Newton's method and LM?
            // W = W - lr*J  // J = Jacobian of the forward pass of one layer
            // J = dC / dw = dC / dy * dy / dw where x is the previous input
            // but its x onwards, y is only the output of the network into the loss
            
            // dC / dy; C = 1/2(label - y)^2
            public static float L2grad(float output, float label)
            {
                return label - output;
            }
            public static float[] L2grad(float[] output, float[] label)
            {
                return MatrixMaths.SubtractMatrices(label, output);
            }
            public static float[][] L2grad(float[][] output, float[][] label)
            {
                return MatrixMaths.SubtractMatrices(label, output);
            }
            // FC jacobian:
            public static float[][] FCgradW(float[][] yPrevious) {
                var prevTranspose = MatrixMaths.Transpose(yPrevious)[0];
                float[][] ret = new float[yPrevious.Length][];
                for (int i = 0;i < yPrevious.Length; i++)
                {
                    ret[i] = prevTranspose;
                }
                return ret;
            }
            // dC / db = dC / dy * dy / db
            // dC / dy already known
            // dy / db: y = kx + b => dy / db = 1
            public static float[][] FCgradB(float[][] yPrevious)
            {
                float [][] ret = new float[yPrevious.Length][];
                for(int i = 0; i < yPrevious.Length; i++)
                {
                    ret[i] = new float[] { 1 };
                }
                return ret;
            }
            // ReLU grad: y = relu(x) => dy / dx = x < 0f ? 0f : 1f;
            public static float ReLUgrad(float x)
            {
                return x < 0f ? 0f : 1f;
            }
            public static float[] ReLUgrad(float[] x)
            {
                float[] ret = new float[x.Length];
                for(int i = 0; i < x.Length; i++)
                {
                    ret[i] = x[i] < 0f ? 0f : 1f;
                }
                return ret;
            }
            public static float[][] ReLUgrad(float[][] x)
            {
                float[][] ret = new float[x.Length][];
                for(int i = 0; i < x.Length; i++)
                {
                    ret[i] = ReLUgrad(x[i]);
                }
                return ret;
            }
            //public static float[][] 
            //public static float SGD() { }
        }
        public class BasicNet
        {
            //float[][] duck = { { 0.2, 0.3 }, { 0.1, 0.5 }, { 0.7, 0.5 } };
            //float[][] yuck = { { 0.1 }, { 0.2 } };
            float[][][] W;
            // [
            //  [[.],[.],[.]],
            //  [[.],[.],[.]],
            //  [[.],[.],[.]],
            //  [[.],[.],[.]],
            // ], shape is (4,3,1): (4 layers, each layer has height 3, and bias is just 1 number that's added to a 3x1 vector), see next liness
            // [
            // [.]
            // [.]
            // [.]
            // ]
            public float[][][] B;
            public float[][] WIH; // weights from Input to first Iidden layer
            public float[][] BI; // Input biases - should be the same height as hidden layer biases
            float[][] WHO; // weights from last Hidden layer to Output
            float[][] BO; // Output biases - should be the same height as the output
            int nLayers;
            int nInputs;
            int nOutputs;
            int hiddenLayerWidths;
            float[] intermediateShape;
            float[][][] intermediateResults;
            float[][][] reluArgs;
            //List<float[][]> argResults = new List<float[][]>();
            float[][][] reluResults;

            public BasicNet(int nInputs, int nOutputs, int nHiddenLayers, int hiddenLayerHeights)
            {
                this.nLayers = nHiddenLayers;
                this.nInputs = nInputs;
                this.nOutputs = nOutputs;
                this.hiddenLayerWidths = hiddenLayerHeights;
                // input height N
                // layer height P
                // first matrix shape P x N; P rows, N columns
                // matrix between hidden layers always square with shape P x P
                // matrix between hidden layer and output layer shape O x P

                // this is P high, N wide
                // [ . . ] [ . ] 2 inputs = [ . ] hidden layer has height 3
                // [ . . ] [ . ]            [ . ]
                // [ . . ]                  [ . ]
                /*
                 input, input into hidden layers * N, output (, input into softmax or other layer, then output)
                //*/
                intermediateResults = new float[1 + 1 + nHiddenLayers + 1][][];  // x, input into 1st hidden, hidden layers*n, input into output
                reluResults = new float[1 + nHiddenLayers + 1][][];  // the above plugged into a relu without the initial x
                intermediateShape = new float[intermediateResults.Length];

                // input to hidden layer
                WIH = MatrixMaths.Init2DMatrix(hiddenLayerHeights, nInputs);
                MatrixMaths.InitializeNormalMatrix(ref WIH);
                BI = MatrixMaths.Init2DMatrix(hiddenLayerHeights, 1);
                MatrixMaths.InitializeNormalMatrix(ref BI);
                                
                //                                   \/ counting the number between hidden layers
                // 1 hidden layer => Input -- Hidden ==> Hidden -- Output
                int heightOfHiddenInputs = hiddenLayerHeights;  // in case of a square network these are the same
                // first argument: number of layers
                // second: height of the inputs to the layer
                // third: number of weights necessary equal to the length of the input vector
                W = MatrixMaths.Init3DMatrix(Math.Max(0, nHiddenLayers - 1), hiddenLayerHeights, heightOfHiddenInputs);
                MatrixMaths.InitializeNormalMatrices(ref W);
                B = MatrixMaths.Init3DMatrix(Math.Max(0, nHiddenLayers - 1), hiddenLayerHeights, 1);
                MatrixMaths.InitializeNormalMatrices(ref B);
                // this is N high O wide
                // [ .  .  . ] [ . ]   [ . ] 2 outputs
                // [ .  .  . ] [ . ] = [ . ]
                //             [ . ]
                WHO = MatrixMaths.Init2DMatrix(nOutputs, hiddenLayerHeights);
                MatrixMaths.InitializeNormalMatrix(ref WHO);
                BO = MatrixMaths.Init2DMatrix(nOutputs, 1);
                MatrixMaths.InitializeNormalMatrix(ref BO);
                //MatrixMaths.PrintMatrix(MatrixMaths.MultiplyMatrix(duck, yuck));
            }
            public void SaveNetwork(int generation)
            {
                using (StreamWriter sw = new StreamWriter("Generation" + generation + ".txt", true))
                {
                    //JsonSerializer serializer = new JsonSerializer();

                    sw.WriteLine(JsonSerializer.Serialize(W));
                    sw.WriteLine(JsonSerializer.Serialize(B));
                    sw.WriteLine(JsonSerializer.Serialize(WIH));
                    sw.WriteLine(JsonSerializer.Serialize(BI));
                    sw.WriteLine(JsonSerializer.Serialize(WHO));
                    sw.WriteLine(JsonSerializer.Serialize(BO));
                }
            }
            public void LoadNetwork(List<object> arrays)
            {

            }
            // forward pass
            public float[][] forward(float[][] x)
            {
                // shape e.g. (3,1)
                // [ [outputnumber1] ]
                // [ [outputnumber2] ]
                // [ [outputnumber3] ]

                // W: 0[[]] 1[[]] 2[[]]
                // Console.WriteLine("x:");
                // MatrixMaths.PrintMatrix(x);
                int intermediateArgsIndex = 0;
                int intermediateReluIndex = 0;
                this.intermediateResults[intermediateArgsIndex++] = x; // save for backprop
                float[][] res = MatrixMaths.MultiplyMatrix(WIH, x);
                // Console.WriteLine("res homogeneous:");
                // MatrixMaths.PrintMatrix(res);
                // MatrixMaths.PrintMatrix(WIH);
                // dont forget biases
                res = MatrixMaths.AddMatrices(BI, res);
                // Console.WriteLine("res bias:");
                // MatrixMaths.PrintMatrix(res);
                // then plug into non-linear fucntion, in our case rn ReLU6
                this.intermediateResults[intermediateArgsIndex++] = res; // o vector as in output
                res = Activations.ReLU6(res); // make all values below zero equal to zero
                                    // Console.WriteLine("res ReLU:");
                                    // MatrixMaths.PrintMatrix(res);
                this.reluResults[intermediateReluIndex++] = x;

                for (int i = 0; i < W.GetLength(0); i++)
                {
                    res = MatrixMaths.MultiplyMatrix(W[i], res);
                    res = MatrixMaths.AddMatrices(B[i], res);
                    this.intermediateResults[intermediateArgsIndex++] = res;
                    res = Activations.ReLU6(res);
                    this.reluResults[intermediateReluIndex++] = x;
                }
                // e.g. [[power_into_left_leg]; [power into right leg]], shape (2,1)
                float[][] output = MatrixMaths.MultiplyMatrix(WHO, res);
                output = MatrixMaths.AddMatrices(BO, res);
                this.intermediateResults[intermediateArgsIndex++] = output;
                //output = Activations.ReLU6(output);
                return output;
            }
            static void AddPerturbation2D(ref float[][] originalMat, float deviation) // height, width of a single weight matrix
            {
                float[][] noise = MatrixMaths.Init2DMatrix(originalMat.GetLength(0), originalMat[0].GetLength(0));
                MatrixMaths.InitializeNormalMatrix(ref noise);
                originalMat = MatrixMaths.AddMatrices(MatrixMaths.MultiplyMatrixByScalar(noise, deviation), originalMat);
            }
            static void AddPerturbation3D(ref float[][][] originalMat, float deviation) // height, width of a single weight matrix
            {
                float[][][] noise = MatrixMaths.Init3DMatrix(originalMat.GetLength(0), originalMat[0].GetLength(0), originalMat[0][0].GetLength(0));
                MatrixMaths.InitializeNormalMatrices(ref noise);
                originalMat = MatrixMaths.AddMatrices3D(MatrixMaths.MultiplyMatricesByScalar(noise, deviation), originalMat);
            }
            public void PerturbateNetwork(float deviation) // usually 0.01
            {
                AddPerturbation2D(ref WIH, deviation);
                AddPerturbation2D(ref BI, deviation);
                AddPerturbation3D(ref W, deviation);
                AddPerturbation3D(ref B, deviation);
                AddPerturbation2D(ref WHO, deviation);
                AddPerturbation2D(ref BO, deviation);
            }
            public void copyWeightsFrom(BasicNet basicNet)
            {
                // copy weights from the "basicNet" argument to "this"
                /*
                    float*** = [float**, float**]
                    float** = [float*, float*]
                    float* = [float, float]
                    typeof(float***) = int64
                    int i = 1;
                    int u = i;
                    // => u = 1;
                    W = 1;
                    other.W = W;
                    => other.W = 1;
                */
                W = ExtensionMethods.DeepClone(basicNet.W);
                B = ExtensionMethods.DeepClone(basicNet.B);
                WIH = ExtensionMethods.DeepClone(basicNet.WIH);
                BI = ExtensionMethods.DeepClone(basicNet.BI);
                WHO = ExtensionMethods.DeepClone(basicNet.WHO);
                BO = ExtensionMethods.DeepClone(basicNet.BO);
            }
        }

    }

    public static class ExtensionMethods
    {
        // Deep clone
        public static T DeepClone<T>(this T a)
        {
            using (MemoryStream stream = new MemoryStream())
            {
                BinaryFormatter formatter = new BinaryFormatter();
                formatter.Serialize(stream, a);
                stream.Position = 0;
                return (T)formatter.Deserialize(stream);
            }
        }
    }
}
