/**
 * @author Dustin Miao
 * @version February 4 2022
 * 
 * An A-B-1 neural network
 * 
 * Methods:
 * - double randomWeight()
 * - double sigmoid(double x)
 * - double dsigmoid(double x)
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

size_t numInputs;          // number of nodes in input layer
double *inputs;            // values of nodes in the input layer
double **inputWeights;     // weights of edges coming from the input layer
double **dinputWeights;    // change in input weights (for training only)

size_t numHiddens;         // number of nodes in hidden layer
double *hiddens;           // values of nodes in the hidden layer
double *hiddenWeights;     // weights of edges coming from the hidden layer
double *dhiddenWeights;    // change in hidden weights (for training only)
double *hiddenThetas;      // values of Theta for nodes in the hidden layer

double output;             // output value
double outputTheta;        // value of Theta for output node

size_t numTestCases;       // number of test cases for training
double **trainInput;       // input for training data
double *trainOutput;       // expected output for training data

double randomWeightMin;    // lower bound for initial randomized weights
double randomWeightMax;    // upper bound for initial randomized weights
size_t maxIterations;      // maximum number of allowed iterations for training
double errorThreshold;     // error threshold to finish training
double learningFactor;     // learning factor 
size_t iteration;          // number of iterations
double maxError;           // maximum error over all test cases

std::string mode;          // current mode (either "train" or "run")

std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count()); // random number generator seeded with current time

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
double sigmoid(double x) 
{
   return 1.0 / (1.0 + exp(-x));
}

/**
 * @brief Calculates the derivative of the sigmoid function at some location.
 * @param x A double.
 * @return The derivative of the sigmoid function at x. 
 */ 
double dsigmoid(double x)
{
   double v = sigmoid(x);
   return v * (1.0 - v);
}

/**
 * @brief Inputs variables from "control.txt"
 * @precondition Nothing.
 * @return Nothing.
 */ 
void configure()
{
   std::ifstream controlFile("control.txt");
   if (controlFile.is_open())
   {
      std::string line;

      // load number of inputs nodes
      std::getline(controlFile, line);
      numInputs = std::stoi(line);

      // load number of hidden nodes
      std::getline(controlFile, line);
      numHiddens = std::stoi(line);

      // gets current mode (train/run)
      std::getline(controlFile, line);
      mode = line;

      if (mode == "train")
      {
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

         // get number of training test cases (necessary to allocate memory)
         std::ifstream traindataFile("./train/traindata.txt");
         if (traindataFile.is_open()) 
         {
            while (getline(traindataFile, line))
            {
               numTestCases++;
            }
            traindataFile.close();
         }
      } // if (mode == "train")

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
    * @precondition numInputs, numHiddens, and numTestCases have already been
    *    initialized to their appropriate values.
    * @return Nothing.
    */ 
   void allocateMemory()
   {
      // allocate memory for input layer
      inputs = new double[numInputs];
      inputWeights = new double*[numInputs];
      dinputWeights = new double*[numInputs];
      for (size_t k = 0; k < numInputs; k++)
      {
         inputWeights[k] = new double[numHiddens];
         dinputWeights[k] = new double[numHiddens];
      }

      // allocate memory for hidden layer
      hiddens = new double[numHiddens];
      hiddenWeights = new double[numHiddens];
      dhiddenWeights = new double[numHiddens];
      hiddenThetas = new double[numHiddens];

      // allocate memory for training data
      trainInput = new double*[numTestCases];
      for (size_t t = 0; t < numTestCases; t++)
      {
         trainInput[t] = new double[numInputs];
      }
      trainOutput = new double[numTestCases];
   } // void allocateMemory()

   /**
    * @brief Randomizes values of weights of both input and hidden layer.
    * @precondition All relevant variables have been initialized and have 
    *    memory allocated for them.
    * @return Nothing. 
    */ 
   void randomizeWeights()
   {
      for (size_t k = 0; k < numInputs; k++)
      {
         for (size_t j = 0; j < numHiddens; j++)
         {
            inputWeights[k][j] = randomWeight();
         }
      }
      for (size_t j = 0; j < numHiddens; j++)
      {
         hiddenWeights[j] = randomWeight();
      }
   } // randomizeWeights()

   /**
    * @brief Loads training data from the file "./train/traindata.txt". 
    * @precondition numInputs and numTestCases have been initialized to their 
    *    appropriate values, and space has been allocated in trainInput and 
    *    trainOutput. The file exists, and it has the number of lines
    *    specified by numTestCases. Each line contains numInputs + 1 floating
    *    point values. The first numInputs numbers specify the input and the 
    *    final number specifies the expected output.
    * @return Nothing. 
    */ 
   void loadData()
   {
      std::ifstream traindataFile("./train/traindata.txt");
      if (traindataFile.is_open())
      {
         std::string line, value;
         for (size_t t = 0; t < numTestCases; t++)
         {
            std::getline(traindataFile, line);
            std::istringstream iss(line);
            for (size_t k = 0; k < numInputs; k++)
            {
               std::getline(iss, value, ' ');
               trainInput[t][k] = std::stod(value);
            }
            std::getline(iss, value, ' ');
            trainOutput[t] = std::stod(value);
         }
         traindataFile.close();
      } // if (traindataFile.is_open())
   } // void loadData()

   /**
    * @brief Prints relevant inputted data into the terminal.
    * @precondition All relevant variables have been initialized and have 
    *    memory allocated for them.
    * @return Nothing. 
    */ 
   void echoData()
   {
      printf("=================================\n");
      printf("Train Mode:\n");
      printf("=================================\n");

      // prints test cases 
      printf("numTestCases = %lu\n", numTestCases);
      for (size_t t = 0; t < numTestCases; t++)
      {
         printf("{");
         for (size_t k = 0; k < numInputs; k++)
         {
            printf("%f", trainInput[t][k]);
            if (k < numInputs - 1)
            {
               printf(", ");
            }
         }
         printf("} expects {%f}\n", trainOutput[t]);
      }
      printf("---------------------------------\n");

      // prints randomized weights
      printf("Randomized Input Weights:\n");
      for (size_t k = 0; k < numInputs; k++)
      {
         for (size_t j = 0; j < numHiddens; j++)
         {
               printf("%lu -> %lu : %f\n", k, j, inputWeights[k][j]);
         }
      }
      printf("---------------------------------\n");
      printf("Randomized Hidden Weights:\n");
      for (size_t j = 0; j < numHiddens; j++)
      {
         printf("%lu -> output : %f\n", j, hiddenWeights[j]);
      }
      printf("---------------------------------\n");

      // prints other input variables
      printf("Runtime Training Parameters:\n");
      printf("  randomWeightMin = %f\n", randomWeightMin);
      printf("  randomWeightMax = %f\n", randomWeightMax);
      printf("  numInputs = %lu\n", numInputs);
      printf("  numHiddens = %lu\n", numHiddens);
      printf("  maxIterations = %lu\n", maxIterations);
      printf("  errorThreshold = %f\n", errorThreshold);
      printf("  learningFactor = %f\n", learningFactor);
      printf("=================================\n");
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
            // run test case through network
            std::copy_n(trainInput[t], numInputs, inputs);
            for (size_t j = 0; j < numHiddens; j++)
            {
               hiddenThetas[j] = 0.0;
               for (size_t k = 0; k < numInputs; k++)
               {
                  hiddenThetas[j] += inputs[k] * inputWeights[k][j];
               }
               hiddens[j] = sigmoid(hiddenThetas[j]);
            }
            outputTheta = 0.0;
            for (size_t j = 0; j < numHiddens; j++)
            {
               outputTheta += hiddens[j] * hiddenWeights[j];
            }
            output = sigmoid(outputTheta);

            // determine error and calculate necessary changes
            double omega = trainOutput[t] - output;
            double psi = omega * dsigmoid(outputTheta);
            for (size_t j = 0; j < numHiddens; j++)
            {
               double partial = -hiddens[j] * psi;
               dhiddenWeights[j] = -learningFactor * partial;
            }

            for (size_t j = 0; j < numHiddens; j++)
            {
               double Omega = psi * hiddenWeights[j];
               double Psi = Omega * dsigmoid(hiddenThetas[j]);
               for (size_t k = 0; k < numInputs; k++)
               {
                  double partial = -inputs[k] * Psi;
                  dinputWeights[k][j] = -learningFactor * partial;
               }
            }

            // apply changes
            for (size_t k = 0; k < numInputs; k++)
            {
               for (size_t j = 0; j < numHiddens; j++)
               {
                  inputWeights[k][j] += dinputWeights[k][j];
               }
            }
            for (size_t j = 0; j < numHiddens; j++)
            {
               hiddenWeights[j] += dhiddenWeights[j];
            }
         } // for (size_t t = 0; t < numTestCases; t++)

         // get maximum error over all test cases
         maxError = 0.0;
         for (size_t t = 0; t < numTestCases; t++)
         {
            // run test case through network
            std::copy_n(trainInput[t], numInputs, inputs);
            for (size_t j = 0; j < numHiddens; j++)
            {
               hiddenThetas[j] = 0.0;
               for (size_t k = 0; k < numInputs; k++)
               {
                  hiddenThetas[j] += inputs[k] * inputWeights[k][j];
               }
               hiddens[j] = sigmoid(hiddenThetas[j]);
            }
            outputTheta = 0.0;
            for (size_t j = 0; j < numHiddens; j++)
            {
               outputTheta += hiddens[j] * hiddenWeights[j];
            }
            output = sigmoid(outputTheta);

            // updates maxError
            double v = trainOutput[t] - output;
            double error = 0.5 * v * v;
            if (maxError < error)
               maxError = error;
         } // for (size_t t = 0; t < numTestCases; t++)

         iteration++;
      } // while (iteration < maxIterations && errorThreshold < maxError)
   } // void trainNetwork() 

   /**
    * @brief Saves the final weight values to "./run/weights.txt"
    * @precondition All relevant variables have been initialized and have 
    *    memory allocated for them.
    * @postcondition The "./run/weights.txt" file has had the weights updated to
    *    their new calculated values. 
    * @return Nothing. 
    */ 
   void saveWeights()
   {
      std::ofstream weightsFile("./run/weights.txt");
      if (weightsFile.is_open())
      {
         for (size_t k = 0; k < numInputs; k++)
         {
            for (size_t j = 0; j < numHiddens; j++)
            {
               weightsFile << inputWeights[k][j] << '\n';
            }
         }
         for (size_t j = 0; j < numHiddens; j++)
         {
            weightsFile << hiddenWeights[j] << '\n';
         }
         weightsFile.close();
      } // if (weightsFile.is_open())
   } // void saveWeights()

   /**
    * @brief Prints to terminal the results of the training, including 
    *    the final weight values and the expected and recieved output for
    *    each test case, including the error. 
    * @precondition All relevant variables have been initialized and have 
    *    memory allocated for them.
    * @return Nothing.
    */ 
   void reportResult()
   {
      // prints data to terminal
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
      printf("\n=================================\n");

      // prints final weight values
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
         printf("    %lu -> output : %f\n", j, hiddenWeights[j]);
      }
      printf("---------------------------------\n");

      // prints test cases, network output, and error
      printf("Truth Table:\n");
      printf("  numTestCases = %lu\n", numTestCases);
      for (size_t t = 0; t < numTestCases; t++)
      {
         std::copy_n(trainInput[t], numInputs, inputs);
         for (size_t j = 0; j < numHiddens; j++)
         {
            hiddenThetas[j] = 0.0;
            for (size_t k = 0; k < numInputs; k++)
            {
               hiddenThetas[j] += inputs[k] * inputWeights[k][j];
            }
            hiddens[j] = sigmoid(hiddenThetas[j]);
         }
         outputTheta = 0.0;
         for (size_t j = 0; j < numHiddens; j++)
         {
            outputTheta += hiddens[j] * hiddenWeights[j];
         }
         output = sigmoid(outputTheta);
         double v = trainOutput[t] - output;
         double error = 0.5 * v * v;

         printf("  %lu. {", t);
         for (size_t k = 0; k < numInputs; k++)
         {
            printf("%f", trainInput[t][k]);
            if (k < numInputs - 1)
            {
               printf(", ");
            }
         }
         printf("} expects {%f}, outputs {%f} with error {%f}\n", trainOutput[t], output, error);
      } // for (size_t t = 0; t < numTestCases; t++)
      printf("=================================\n");
   } // void reportResult()
} // namespace::train

/**
 * Contains the relevant methods to run the neural network
 */
namespace run
{
   /**
    * @brief Allocates the necessary memory for running the neural network.
    * @precondition numInputs and numHiddens have already been initialized 
    *    to their appropriate values.
    * @return Nothing.
    */ 
   void allocateMemory()
   {
      inputs = new double[numInputs];
      inputWeights = new double*[numInputs];
      for (size_t k = 0; k < numInputs; k++)
      {
         inputWeights[k] = new double[numHiddens];
      }

      hiddens = new double[numHiddens];
      hiddenWeights = new double[numHiddens];
      hiddenThetas = new double[numHiddens];
   } // void allocateMemory()

   /**
    * @brief Loads input data from the file "./run/inputs.txt", 
    *    and loads weight data from the file "./run/weights.txt".
    * @precondition All relevant variables have been initialized and have 
    *    memory allocated for them. The inputs file exists, and it must have
    *    exactly numInputs number of lines, each line with a single value.
    *    The weights file exists, and it must contain numInputs * numHiddens
    *    + numHiddens values, one on each line. 
    * @return Nothing. 
    */ 
   void loadData()
   {
      // load inputs
      std::ifstream inputsFile("./run/inputs.txt");
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
      std::ifstream weightsFile("./run/weights.txt");
      if (weightsFile.is_open())
      {
         std::string line;
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
            std::getline(weightsFile, line);
            hiddenWeights[j] = std::stod(line);
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
      printf("=================================\n");
      printf("Run Mode:\n");
      printf("=================================\n");

      // prints inputs
      printf("Inputs:\n");
      for (size_t k = 0; k < numInputs; k++)
      {
         printf("  %f\n", inputs[k]);
      }
      printf("---------------------------------\n");

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
         printf("    %lu -> output : %f\n", j, hiddenWeights[j]);
      }
      printf("---------------------------------\n");

      // prints other variables
      printf("Parameters:\n");
      printf("  numInputs = %lu\n", numInputs);
      printf("  numHiddens = %lu\n", numHiddens);
      printf("=================================\n");
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
         hiddenThetas[j] = 0.0;
         for (size_t k = 0; k < numInputs; k++)
         {
            hiddenThetas[j] += inputs[k] * inputWeights[k][j];
         }
         hiddens[j] = sigmoid(hiddenThetas[j]);
      }
      outputTheta = 0.0;
      for (size_t j = 0; j < numHiddens; j++)
      {
         outputTheta += hiddens[j] * hiddenWeights[j];
      }
      output = sigmoid(outputTheta);
   } // void runNetwork()

   /**
    * @brief Prints the output value of the neural network to the terminal.
    * @return Nothing. 
    */ 
   void reportResult()
   {
      printf("Output = %f\n", output);
      printf("=================================\n");
   }
} // namespace::run

/**
 * @brief Executes the neural network with instructions specified in "control.txt".
 * @precondition "control.txt" must exist, and hold the following information, one 
 *    per line, in the following order: numInputs(positive integer), numHiddens(positive 
 *    integer), mode("train"/"run"). If the mode is "train", then the file should 
 *    also contain: randomWeightMin(a double), randomWeightMax(a double greater than 
 *    randomWeightMin), maxIterations(a positive integer), errorThreshold(a non-negative
 *    real), and learningFactor(a positive double). 
 */ 
int main() 
{
   configure();

   if (mode == "train")
   {
      train::allocateMemory();
      train::randomizeWeights();
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
} // int main()