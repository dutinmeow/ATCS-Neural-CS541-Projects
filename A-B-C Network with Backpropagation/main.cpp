/**
 * @author Dustin Miao
 * @version February 28 2022
 *
 * An A-B-C neural network with file input/output and backpropagation
 *
 * Methods:
 * - double randomWeight()
 * - double activationFunction(double x)
 * - double derivActivationFunction(double x)
 *
 * - void train::allocateMemory()
 * - void train::randomizeWeights()
 * - void train::loadData()
 * - void train::echoData()
 * - void train::trainNetwork()
 * - void train::reportResult()
 *
 * - void run::allocateMemory()
 * - void run::loadData()
 * - void run::echoData()
 * - void run::runNetwork()
 * - void run::reportResult()
 */

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <sstream>

#pragma region declarations

size_t numInputs;             // number of nodes in input layer
double *inputs;               // values of nodes in the input layer
double **inputWeights;        // weights of edges coming from the input layer

size_t numHiddens;            // number of nodes in hidden layer
double *hiddens;              // values of nodes in the hidden layer
double **hiddenWeights;       // weights of edges coming from the hidden layer
double *hiddenThetas;         // values of Theta for nodes in the hidden layer (for training only)

size_t numOutputs;            // number of nodes in output layer
double *outputs;              // output value
double *psi;                  // value of psi for output nodes (for training only)

size_t numTestCases;          // number of test cases for training
double **trainInput;          // input for training data
double **trainOutput;         // expected outputs for training data

bool loadWeightsFromFile;     // true if load weights from file, false if randomly generate
double randomWeightMin;       // lower bound for initial randomized weights
double randomWeightMax;       // upper bound for initial randomized weights
size_t maxIterations;         // maximum number of allowed iterations for training
double errorThreshold;        // error threshold to finish training
double learningFactor;        // learning factor
size_t iteration;             // number of iterations
double maxError;              // maximum error over all test cases

std::string mode;               // current mode (either "train" or "run")
std::string controlFileName;  // relative path of control file
std::string trainFileName;    // relative path of train data file (for training only)
std::string inputsFileName;   // relative path of inputs file (for running only)
std::string weightsFileName;  // relative path of weights file (for both training and running)

std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count()); // random number generator seeded with current time

#pragma endregion declaration

/**
 * @brief Generates a random number between randomWeightMin and randomWeightMax with
 * a uniform distribution.
 * @return A double value in the range [randomWeightMin, randomWeightMax].
 */
double randomWeight()
{
   return std::uniform_real_distribution<double>(randomWeightMin, randomWeightMax)(rng);
}

/**
 * @brief Calculates the sigmoid function at some location.
 * @param x A double.
 * @return The sigmoid function at x.
 */
double activationFunction(double x)
{
   return 1.0 / (1.0 + exp(-x));
}

/**
 * @brief Calculates the derivative of the sigmoid function at some location.
 * @param x A double.
 * @return The derivative of the sigmoid function at x.
 */
double derivActivationFunction(double x)
{
   double v = activationFunction(x);
   return v * (1.0 - v);
}

/**
 * @brief Inputs variables from "control.txt"
 * @precondition "control.txt" exists and has the correct data.
 * @return Nothing.
 */
void configure()
{
   // load control information
   std::ifstream controlFile(controlFileName);

   if (controlFile.is_open())
   {
      std::string line;

      // load number of inputs nodes
      std::getline(controlFile, line);
      numInputs = std::stoi(line);

      // load number of hidden nodes
      std::getline(controlFile, line);
      numHiddens = std::stoi(line);

      // load number of output nodes
      std::getline(controlFile, line);
      numOutputs = std::stoi(line);

      // gets current mode (train/run)
      std::getline(controlFile, line);
      mode = line;

      if (mode == "train")
      {
         // load train file name
         std::getline(controlFile, line);
         trainFileName = line;

         // load weights file name
         std::getline(controlFile, line);
         weightsFileName = line;

         std::getline(controlFile, line);
         loadWeightsFromFile = bool(stoi(line));

         // load random weight minimum
         std::getline(controlFile, line);
         randomWeightMin = std::stod(line);

         // load random weight maximum
         std::getline(controlFile, line);
         randomWeightMax = std::stod(line);

         assert(randomWeightMin <= randomWeightMax);

         // load maximum iterations
         std::getline(controlFile, line);
         maxIterations = std::stoi(line);

         // load error threshold
         std::getline(controlFile, line);
         errorThreshold = std::stod(line);

         // load learning factor
         std::getline(controlFile, line);
         learningFactor = std::stod(line);

         // load number of training test cases (necessary to allocate memory)
         std::ifstream trainFile(trainFileName);

         if (trainFile.is_open())
         {
            while (getline(trainFile, line))
            {
               numTestCases++;
            }

            trainFile.close();
         }
      } // if (mode == "train")
      else if (mode == "run")
      {
         std::getline(controlFile, line);
         inputsFileName = line;

         std::getline(controlFile, line);
         weightsFileName = line;
      }

      controlFile.close();
   } // if (controlFile.is_open())
} // void configure()

/**
 * @brief Contains relevant methods to train the neural network.
 */
namespace train
{
   /**
    * @brief Allocates the necessary memory for training the neural network.
    * @precondition numInputs, numHiddens, numOutputs, and numTestCases have
    *    already been initialized to their appropriate values.
    * @return Nothing.
    */
   void allocateMemory()
   {
      // allocate memory for input layer
      inputs = new double[numInputs];
      inputWeights = new double*[numInputs];

      // allocate memory for input weights
      for (size_t k = 0; k < numInputs; k++)
      {
         inputWeights[k] = new double[numHiddens];
      }

      // allocate memory for hidden layer
      hiddens = new double[numHiddens];
      hiddenWeights = new double*[numHiddens];
      hiddenThetas = new double[numHiddens];

      // allocate memory for hidden weights
      for (size_t j = 0; j < numHiddens; j++)
      {
         hiddenWeights[j] = new double[numOutputs];
      }

      // allocate memory for output layer
      outputs = new double[numOutputs];
      psi = new double[numOutputs];

      // allocate memory for training data
      trainInput = new double*[numTestCases];
      trainOutput = new double*[numTestCases];

      // allocate memory for all test cases
      for (size_t t = 0; t < numTestCases; t++)
      {
         trainInput[t] = new double[numInputs];
         trainOutput[t] = new double[numOutputs];
      }
   } // void allocateMemory()

   /**
    * @brief Randomizes values of weights of both input and hidden layer.
    * @precondition All relevant variables have been initialized and have
    *    memory allocated for them.
    * @return Nothing.
    */
   void randomizeWeights()
   {
      // iterate through all input weights
      for (size_t k = 0; k < numInputs; k++)
      {
         for (size_t j = 0; j < numHiddens; j++)
         {
            inputWeights[k][j] = randomWeight();
         }
      }

      // iterate through all hidden weights
      for (size_t j = 0; j < numHiddens; j++)
      {
         for (size_t i = 0; i < numOutputs; i++)
         {
            hiddenWeights[j][i] = randomWeight();
         }
      }
   } // randomizeWeights()

   /**
    * @brief Loads weights from file of both input and hidden layer.
    * @precondition All relevant variables have been initialized and have
    *    memory allocated for them.
    * @return Nothing.
    */ 
   void loadWeights() 
   {
      // load weights
      std::ifstream weightsFile(weightsFileName);

      if (weightsFile.is_open())
      {
         std::string line;

         std::getline(weightsFile, line);
         assert(stoi(line) == (int)numInputs);

         std::getline(weightsFile, line);
         assert(stoi(line) == (int)numHiddens);

         std::getline(weightsFile, line);
         assert(stoi(line) == (int)numOutputs);

         for (size_t k = 0; k < numInputs; k++)
         {
            for (size_t j = 0; j < numHiddens; j++)
            {
               std::getline(weightsFile, line);
               inputWeights[k][j] = std::stod(line);
            }
         }

         for (size_t j = 0; j < numHiddens; j++)
         {
            for (size_t i = 0; i < numOutputs; i++)
            {
               std::getline(weightsFile, line);
               hiddenWeights[j][i] = std::stod(line);
            }
         }

         weightsFile.close();
      } // if (weightsFile.is_open())
   }

   /**
    * @brief Loads training data from the file trainFileName, which was to 
    *   be specified within the control file. 
    * @precondition numInputs and numTestCases have been initialized to their
    *    appropriate values, and space has been allocated in trainInput and
    *    trainOutput. The file exists, and it has the number of lines
    *    specified by numTestCases. Each line contains numInputs + numOutputs
    *    floating point values. The first numInputs numbers specify the
    *    input and the next numOutputs specifies the corresponding outputs.
    * @return Nothing.
    */
   void loadData()
   {
      // load training data
      std::ifstream trainFile(trainFileName);

      if (trainFile.is_open())
      {
         std::string line, value;

         for (size_t t = 0; t < numTestCases; t++)
         {
            std::getline(trainFile, line);
            std::istringstream iss(line);

            // read inputs for current test case
            for (size_t k = 0; k < numInputs; k++)
            {
               std::getline(iss, value, ' ');
               trainInput[t][k] = std::stod(value);
            }

            // read expected outputs for current test case
            for (size_t i = 0; i < numOutputs; i++)
            {
               std::getline(iss, value, ' ');
               trainOutput[t][i] = std::stod(value);
            }
         }

         trainFile.close();
      } // if (trainFile.is_open())
   } // void loadData()

   /**
    * @brief Prints relevant inputted data into the terminal.
    * @precondition All relevant variables have been initialized and have
    *    memory allocated for them.
    * @return Nothing.
    */
   void echoData()
   {
      printf("==================================================================\n");
      printf("Train Mode:\n");
      printf("==================================================================\n");

      // print file locations
      printf("control file path = \"%s\"\n", controlFileName.c_str());
      printf("train file path = \"%s\"\n", trainFileName.c_str());
      printf("weights file path = \"%s\"\n", weightsFileName.c_str());

      printf("==================================================================\n");

      // prints test cases
      printf("numTestCases = %lu\n", numTestCases);
      for (size_t t = 0; t < numTestCases; t++)
      {
         printf("{");

         // print inputs for current test case
         for (size_t k = 0; k < numInputs; k++)
         {
            printf("%f", trainInput[t][k]);
            if (k < numInputs - 1) printf(", ");
         }

         printf("} expects {");

         // print corresponding outputs for current test case
         for (size_t i = 0; i < numOutputs; i++)
         {
            printf("%f", trainOutput[t][i]);
            if (i < numOutputs - 1) printf(", ");
         }

         printf("}\n");
      } // for (size_t t = 0; t < numTestCases; t++)

      printf("------------------------------------------------------------------\n");

      // print randomized weights
      printf("Weights:\n");
      printf("  Input Weights:\n");

      for (size_t k = 0; k < numInputs; k++)
      {
         for (size_t j = 0; j < numHiddens; j++)
         {
            printf("    %lu -> %lu : %f\n", k, j, inputWeights[k][j]);
         }
      }

      printf("\n");
      printf("  Hidden Weights:\n");

      for (size_t j = 0; j < numHiddens; j++)
      {
         for (size_t i = 0; i < numOutputs; i++)
         {
            printf("    %lu -> %lu : %f\n", j, i, hiddenWeights[j][i]);
         }
      }

      printf("------------------------------------------------------------------\n");
         
      // print other training parameters
      printf("Runtime Training Parameters:\n");
      printf("  loadWeightsFromFile = %d\n", loadWeightsFromFile);
      printf("  randomWeightMin = %f\n", randomWeightMin);
      printf("  randomWeightMax = %f\n", randomWeightMax);
      printf("  numInputs = %lu\n", numInputs);
      printf("  numHiddens = %lu\n", numHiddens);
      printf("  numOutputs = %lu\n", numOutputs);
      printf("  maxIterations = %lu\n", maxIterations);
      printf("  errorThreshold = %f\n", errorThreshold);
      printf("  learningFactor = %f\n", learningFactor);

      printf("==================================================================\n");
   } // void echoData()

   /**
    * @brief Configures the network to minimize error based off of the
    *    randomized initial weight values and the provided training data.
    * @precondition All relevant variables have been initialized and have
    *    memory allocated for them.
    * @postcondition All weights have been updated to their final values.
    * @return Nothing.
    */
   void trainNetwork()
   {
      iteration = 0;
      maxError = std::numeric_limits<double>::max();

      while (iteration < maxIterations && errorThreshold < maxError)
      {
         for (size_t t = 0; t < numTestCases; t++)
         {
            // copy current test case inputs into network
            std::copy_n(trainInput[t], numInputs, inputs);

            // run test case through network and update psi array
            for (size_t j = 0; j < numHiddens; j++)
            {
               hiddenThetas[j] = 0.0;
               for (size_t k = 0; k < numInputs; k++)
               {
                  hiddenThetas[j] += inputs[k] * inputWeights[k][j];
               }

               hiddens[j] = activationFunction(hiddenThetas[j]);
            }

            for (size_t i = 0; i < numOutputs; i++)
            {
               double outputTheta = 0.0;
               for (size_t j = 0; j < numHiddens; j++)
               {
                  outputTheta += hiddens[j] * hiddenWeights[j][i];
               }

               outputs[i] = activationFunction(outputTheta);
               psi[i] = (trainOutput[t][i] - outputs[i]) * derivActivationFunction(outputTheta);
            }

            for (size_t j = 0; j < numHiddens; j++)
            {
               double Omega = 0.0;
               for (size_t i = 0; i < numOutputs; i++)
               {
                  Omega += psi[i] * hiddenWeights[j][i];
                  hiddenWeights[j][i] += learningFactor * hiddens[j] * psi[i];
               }

               double Psi = Omega * derivActivationFunction(hiddenThetas[j]);
               for (size_t k = 0; k < numInputs; k++)
               {
                  inputWeights[k][j] += learningFactor * inputs[k] * Psi;
               }
            } // for (size_t j = 0; j < numHiddens; j++)
         } // for (size_t t = 0; t < numTestCases; t++)

         maxError = 0.0;

         // get maximum error over all test cases
         for (size_t t = 0; t < numTestCases; t++)
         {
            // copy current test case inputs into network
            std::copy_n(trainInput[t], numInputs, inputs);

            // run test case through network
            for (size_t j = 0; j < numHiddens; j++)
            {
               hiddenThetas[j] = 0.0;
               for (size_t k = 0; k < numInputs; k++)
               {
                  hiddenThetas[j] += inputs[k] * inputWeights[k][j];
               }
               hiddens[j] = activationFunction(hiddenThetas[j]);
            }

            for (size_t i = 0; i < numOutputs; i++)
            {
               double outputTheta = 0.0;
               for (size_t j = 0; j < numHiddens; j++)
               {
                  outputTheta += hiddens[j] * hiddenWeights[j][i];
               }
               outputs[i] = activationFunction(outputTheta);
            }

            // calculates error
            double error = 0.0;
            for (size_t i = 0; i < numOutputs; i++)
            {
               double v = trainOutput[t][i] - outputs[i];
               error += v * v;
            }
            error *= 0.5;

            // updates maximum error
            if (maxError < error) maxError = error;
         } // for (size_t t = 0; t < numTestCases; t++)

         iteration++;
      } // while (iteration < maxIterations && errorThreshold < maxError)
   } // void trainNetwork()

   /**
    * @brief Saves the final weight values to weightsFileName, which was 
    *    to be specified within the control file. 
    * @precondition All relevant variables have been initialized and have
    *    memory allocated for them.
    * @postcondition The file located at weightsFileName 
    *    has had the weights updated to their new calculated values.
    * @return Nothing.
    */
   void saveWeights()
   {
      // write to output file
      std::ofstream weightsFile(weightsFileName);

      if (weightsFile.is_open())
      {
         // store training parameters
         weightsFile << numInputs << '\n' << numHiddens << '\n' << numOutputs << '\n';

         for (size_t k = 0; k < numInputs; k++)
         {
            for (size_t j = 0; j < numHiddens; j++)
            {
               weightsFile << inputWeights[k][j] << '\n';
            }
         }

         for (size_t j = 0; j < numHiddens; j++)
         {
            for (size_t i = 0; i < numOutputs; i++)
            {
               weightsFile << hiddenWeights[j][i] << '\n';
            }
         }

         weightsFile.close();
      } // if (weightsFile.is_open())
   } // void saveWeights()

   /**
    * @brief Prints to terminal the results of the training, including
    *    the final weight values and the expected and received outputs for
    *    each test case and the error associated with that test case.
    * @precondition All relevant variables have been initialized and have
    *    memory allocated for them.
    * @return Nothing.
    */
   void reportResult()
   {
      // print runtime training data to the terminal
      printf("Training Exit Information:\n");
      printf("  iterations: %lu\n", iteration);
      printf("  maxError = %f\n", maxError);
      printf("  Termination Condition: ");

      if (maxIterations <= iteration)
      {
         printf("max iterations exceeded, ");
      }

      if (maxError < errorThreshold)
      {
         printf("error threshold reached, ");
      }

      printf("\n==================================================================\n");

      /*
      // prints final calculated weight values
      printf("Weights:\n");
      printf("  Input Weights:\n");

      for (size_t k = 0; k < numInputs; k++)
      {
         for (size_t j = 0; j < numHiddens; j++)
         {
            printf("    %lu -> %lu : %f\n", k, j, inputWeights[k][j]);
         }
      }

      printf("\n");
      printf("  Hidden Weights:\n");

      for (size_t j = 0; j < numHiddens; j++)
      {
         for (size_t i = 0; i < numOutputs; i++)
         {
            printf("    %lu -> %lu : %f\n", j, i, hiddenWeights[j][i]);
         }
      }

      printf("------------------------------------------------------------------\n");
      */

      // prints test cases, network output, and error
      printf("Truth Table:\n");
      printf("  numTestCases = %lu\n", numTestCases);
      for (size_t t = 0; t < numTestCases; t++)
      {
         // copy current test case inputs into network
         std::copy_n(trainInput[t], numInputs, inputs);

         // run test case through network
         for (size_t j = 0; j < numHiddens; j++)
         {
            hiddenThetas[j] = 0.0;
            for (size_t k = 0; k < numInputs; k++)
            {
               hiddenThetas[j] += inputs[k] * inputWeights[k][j];
            }
            hiddens[j] = activationFunction(hiddenThetas[j]);
         }

         for (size_t i = 0; i < numOutputs; i++)
         {
            double outputTheta = 0.0;
            for (size_t j = 0; j < numHiddens; j++)
            {
               outputTheta += hiddens[j] * hiddenWeights[j][i];
            }
            outputs[i] = activationFunction(outputTheta);
         }

         double error = 0.0;

         // calculates error and updates maxError
         for (size_t i = 0; i < numOutputs; i++)
         {
            double v = trainOutput[t][i] - outputs[i];
            error += v * v;
         }

         error *= 0.5;
         printf("  %lu {", t);

         for (size_t k = 0; k < numInputs; k++)
         {
            printf("%f", trainInput[t][k]);
            if (k < numInputs - 1) printf(", ");
         }

         printf("} expects {");

         for (size_t i = 0; i < numOutputs; i++)
         {
            printf("%f", trainOutput[t][i]);
            if (i < numOutputs - 1) printf(", ");
         }

         printf("}, outputs {");

         for (size_t i = 0; i < numOutputs; i++)
         {
            printf("%f", outputs[i]);
            if (i < numOutputs - 1) printf(", ");
         }

         printf("} with error {%f}\n", error);
      } // for (size_t t = 0; t < numTestCases; t++)

      printf("==================================================================\n");
   } // void reportResult()
} // namespace::train

/**
 * Contains the relevant methods to run the neural network, assuming that 
 * it already has been trained. 
 */
namespace run
{
   /**
    * @brief Allocates the necessary memory for running the neural network.
    * @precondition numInputs, numHiddens, and numOutputs have already been initialized
    *    to their appropriate values.
    * @return Nothing.
    */
   void allocateMemory()
   {
      // allocate memory for input layer
      inputs = new double[numInputs];
      inputWeights = new double*[numInputs];

      // allocate memory for input weights
      for (size_t k = 0; k < numInputs; k++)
      {
         inputWeights[k] = new double[numHiddens];
      }

      // allocate memory for hidden layer
      hiddens = new double[numHiddens];
      hiddenWeights = new double*[numHiddens];

      // allocate memory for hidden weights
      for (size_t j = 0; j < numHiddens; j++)
      {
         hiddenWeights[j] = new double[numOutputs];
      }

      // allocate memory for output layer
      outputs = new double[numOutputs];
   } // void allocateMemory()

   /**
    * @brief Loads input data from the file inputsFileName 
    *    and loads weight data from the file weightsFileName, both of which
    *    are to be specified in the control file. 
    * @precondition All relevant variables have been initialized and have
    *    memory allocated for them. The inputs file exists, and it must have
    *    exactly numInputs number of lines, each line with a single value.
    *    The weights file exists, and it must contain numInputs * numHiddens
    *    + numHiddens * numOutputs values, one on each line.
    * @return Nothing.
    */
   void loadData()
   {
      // load inputs
      std::ifstream inputsFile(inputsFileName);

      if (inputsFile.is_open())
      {
         std::string line;

         for (size_t k = 0; k < numInputs; k++)
         {
            std::getline(inputsFile, line);
            inputs[k] = std::stod(line);
         }

         inputsFile.close();
      }

      // load weights
      std::ifstream weightsFile(weightsFileName);

      if (weightsFile.is_open())
      {
         std::string line;

         std::getline(weightsFile, line);
         assert(stoi(line) == (int)numInputs);

         std::getline(weightsFile, line);
         assert(stoi(line) == (int)numHiddens);

         std::getline(weightsFile, line);
         assert(stoi(line) == (int)numOutputs);

         for (size_t k = 0; k < numInputs; k++)
         {
            for (size_t j = 0; j < numHiddens; j++)
            {
               std::getline(weightsFile, line);
               inputWeights[k][j] = std::stod(line);
            }
         }

         for (size_t j = 0; j < numHiddens; j++)
         {
            for (size_t i = 0; i < numOutputs; i++)
            {
               std::getline(weightsFile, line);
               hiddenWeights[j][i] = std::stod(line);
            }
         }

         weightsFile.close();
      } // if (weightsFile.is_open())
   } // void loadData()

   /**
    * @brief Prints relevant inputted data into the terminal.
    * @precondition All relevant variables have been initialized and have
    *    memory allocated for them.
    * @return Nothing.
    */
   void echoData()
   {
      printf("==================================================================\n");
      printf("Run Mode:\n");
      printf("==================================================================\n");

      // print file locations
      printf("control file path = \"%s\"\n", controlFileName.c_str());
      printf("inputs file path = \"%s\"\n", inputsFileName.c_str());
      printf("weights file path = \"%s\"\n", weightsFileName.c_str());

      printf("==================================================================\n");

      // prints inputs
      printf("Inputs:\n");

      for (size_t k = 0; k < numInputs; k++)
      {
         printf("  %f\n", inputs[k]);
      }

      printf("------------------------------------------------------------------\n");

      /*
      // prints weights
      printf("Weights:\n");
      printf("  Input Weights:\n");

      for (size_t k = 0; k < numInputs; k++)
      {
         for (size_t j = 0; j < numHiddens; j++)
         {
            printf("    %lu -> %lu : %f\n", k, j, inputWeights[k][j]);
         }
      }

      printf("\n");
      printf("  Hidden Weights:\n");

      for (size_t j = 0; j < numHiddens; j++)
      {
         for (size_t i = 0; i < numOutputs; i++)
         {
            printf("    %lu -> %lu : %f\n", j, i, hiddenWeights[j][i]);
         }
      }

      printf("------------------------------------------------------------------\n");
      */

      // prints other runnning pararmeters
      printf("Parameters:\n");
      printf("  numInputs = %lu\n", numInputs);
      printf("  numHiddens = %lu\n", numHiddens);
      printf("  numOutputs = %lu\n", numOutputs);

      printf("==================================================================\n");
   } // void echoData()

   /**
    * @brief Runs the input data through the network.
    * @precondition All relevant variables have been initialized and have
    *    memory allocated for them.
    * @postcondition hiddens and output now hold the updated versions of
    *    the data based on the weights and the inputs. The weights and
    *    inputs remain unchanged.
    * @return Nothing.
    */
   void runNetwork()
   {
      for (size_t j = 0; j < numHiddens; j++)
      {
         double hiddenTheta = 0.0;
         for (size_t k = 0; k < numInputs; k++)
         {
            hiddenTheta += inputs[k] * inputWeights[k][j];
         }
         hiddens[j] = activationFunction(hiddenTheta);
      }

      for (size_t i = 0; i < numOutputs; i++)
      {
         double outputTheta = 0.0;
         for (size_t j = 0; j < numHiddens; j++)
         {
            outputTheta += hiddens[j] * hiddenWeights[j][i];
         }
         outputs[i] = activationFunction(outputTheta);
      }
   } // void runNetwork()

   /**
    * @brief Prints the output value of the neural network to the terminal.
    * @return Nothing.
    */
   void reportResult()
   {
      printf("Output = {");

      for (size_t i = 0; i < numOutputs; i++)
      {
         printf("%f", outputs[i]);
         if (i < numOutputs - 1) printf(", ");
      }
      printf("}\n");

      printf("==================================================================\n");
   }
} // namespace::run

/**
 * @brief Executes the neural network with instructions specified in a file passed
 *    through the command line. 
 * @precondition If a control file is passed through the command line (in arguemnt 1), it is
 *    to hold the following information, one per line, in the following order: 
 *    numInputs(positive integer), numHiddens(positive integer), mode("train"/"run"). 
 *    If the mode is "train", then the file should also contain: randomWeightMin(a double),
 *    randomWeightMax(a double greater than randomWeightMin), maxIterations
 *    (a positive integer), errorThreshold(a non-negative real), and learningFactor
 *    (a positive double).
 */
int main(int argc, char *argv[])
{
   // get control file from command line
   if (argc == 1) controlFileName = "control.txt"; // default control file name
   else controlFileName = argv[1];                 // second arguemnt (index 1 of argv) should be file path of control file

   auto startTime = std::chrono::high_resolution_clock::now(); // start timer

   configure();

   if (mode == "train")
   {
      train::allocateMemory();
      if (loadWeightsFromFile) train::loadWeights();
      else train::randomizeWeights();
      train::loadData();
      train::echoData();
      train::trainNetwork();
      train::saveWeights();
      train::reportResult();
   }
   else if (mode == "run")
   {
      run::allocateMemory();
      run::loadData();
      run::echoData();
      run::runNetwork();
      run::reportResult();
   }

   auto endTime = std::chrono::high_resolution_clock::now(); // end timer
   printf("Time Taken: %lu microseconds\n", static_cast<unsigned long>(
      std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count()));
   printf("==================================================================\n");
} // int main()