/***************************************************************************
 *   Copyright (C) 2008 by Orivaldo Vieira Santana Jr   *
 *      *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <cstdlib>
#include <time.h>
#include <sstream>
#include <cmath>

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

#define WEIGHT_INITIALIZATION ((rand()*2 - 1 )%30)/100.0 

using namespace std;

class Neuron;

string int2string( int i )
{
   std::string s;
   std::stringstream out;
   out << i;
   return out.str();
}

class Weight
{
   // fazer um gatNeuronOutput dentro da clase Weight
   private:
      double value;
      Neuron* p_origin;
      Neuron* p_target;
      string label;

   public:
      Weight(double v = 0, Neuron* o = NULL, Neuron* t = NULL, string l = "w ")
      {
         value = v;
         p_origin = o;
         p_target = t;
         label = l;
      }
      Neuron* getAheadNeuron() { return p_target; }
      Neuron* getBackNeuron() { return p_origin; }
      string getLabel() { return label; }
      double getValue() { return value; }
      void setValue(double v) { value = v; }
      void print()
      {
         cout << value << " = " << label;
      }
};

// - a taxa de aprendizagem do neurônio e configurada como um paramentro da rede
// - o neurônio armazena seu valor de saída toda vez que uma entrada é apresentada a rede
class Neuron
{
   private:
      double bias;
      vector<Weight*> inputWieghts;
      vector<Weight*> outputWieghts;
      string label;
      string description;
      double net;
      double output;
      double localGradient;
   public:



      Neuron(string l = "n ", string d = "d ", double b = 1 )
      {
         label = l;
         description = d;
         bias = b;
         localGradient = 0;
      }
      void addInputWeight(Weight* iw)
      {
         inputWieghts.push_back(iw);
      }
      void addOutputWeight(Weight* ow)
      {
         outputWieghts.push_back(ow);
      }
      double getLocalGradient()
      {
         return localGradient;
      }
      string getLabel() 
      {
         return label; 
      }
      double getOutput()
      {
         return output;
      }
      double thrasholdedOutput()
      {
         if (output > 0)
            output = 1;
         else
            output = -1;
         return output;
      }
      double calculateNet(vector<double> *input)
      {
         // posso melhorar este metodo pegando o valor do neuronio a partir do ponteiro para o peso
         vector<Weight*>::iterator itW;
         itW = inputWieghts.begin();
         if (inputWieghts.empty())
            return 0;
         net = (*itW)->getValue() * bias;
         //cout << endl << bias << " * " << (*itW)->getValue();
         itW++;
         vector<double>::iterator itI;
         itI = input->begin();
         while ((itI != input->end()) and (itW != inputWieghts.end()) )
         {
            //cout <<" + " << (*itI) << " * " << (*itW)->getValue();
            net = net + (*itW)->getValue() * (*itI);
            itW++;
            itI++;
         }
         //cout << endl << "net = " << net;
         return net;
      }
      double activationFunctionDerivate()
      {
         //return output*(1-output)
         return pow(cosh(net),-2);
      }
      double calculateOutput(vector<double> *input)
      {
         // sigmoidal activation function 
         //output = 1/(1 + exp(-calculateNet(input)));
         calculateNet(input);
         output = (exp(net) - exp(-net))/(exp(net) + exp(-net));
         //cout << ", y = " << output;
         return output;
      }
      double calculateLocalGradientOfOutputNeuron(double const &d)
      {
         // local gradient = erro * derivate of ativation function
         return (d - output)*activationFunctionDerivate();
      }
      double calculateLocalGradientOfHidenNeuron()
      {
         // to use this function must be first calculate the output and the gradient ahead
         // local gradient = sum weighted of ahead local gradient * derivate of ativation function
         double sum = 0;
         Neuron* neuronK = NULL;
         for (int k = 0; k < outputWieghts.size(); k++)
         {
            if (outputWieghts[k] == NULL)
               continue;
            neuronK = outputWieghts[k]->getAheadNeuron();
            sum = sum + outputWieghts[k]->getValue()*neuronK->getLocalGradient();
         }
         return sum * activationFunctionDerivate();
      }
      void updateHidenWeights(double const &eta, vector<double> *in)
      {
         // preciso garantir que a saida ja tenha sido calculada antes
         localGradient = calculateLocalGradientOfHidenNeuron();
         updateWeights(eta,in);
      }
      void updateOutputWeights(double const &eta, double const &d, vector<double> *in)
      {
         localGradient = calculateLocalGradientOfOutputNeuron(d);
         updateWeights(eta,in);
      }

      void updateWeights(double const &eta, vector<double> *in) 
      {

         // o primeiro peso, de indice 0, esta relacionado ao bias
         inputWieghts[0]->setValue(inputWieghts[0]->getValue() + eta*localGradient*bias  );
         for (int i = 0; i < inputWieghts.size()-1; i++)
         {
            inputWieghts[i+1]->setValue(inputWieghts[i+1]->getValue() + eta*localGradient*in->at(i));
         }
      }

      void clear()
      {
         vector<Weight*>::iterator itW;
         itW = inputWieghts.begin();
         while (itW != inputWieghts.end())
         {
            delete (*itW);
            itW++;
         }
         inputWieghts.clear();
         itW = outputWieghts.begin();
         while (itW != outputWieghts.end())
         {
            delete (*itW);
            itW++;
         }
         outputWieghts.clear();
      }

      void print()
      {
         vector<Weight*>::iterator itW;
         itW = inputWieghts.begin();
         cout << description << ":";
         while (itW != inputWieghts.end())
         {
            cout << " " << (*itW)->getLabel() << "(" << (*itW)->getValue() << ")";
            itW++;
         }
         cout << " <- " << label << " -> ";
         itW = outputWieghts.begin();
         while (itW != outputWieghts.end())
         {
            cout << " " << (*itW)->getLabel() << "(" << (*itW)->getValue() << ")" ;
            itW++;
         }
         cout << endl;
      }
      void simplePrint()
      {
         vector<Weight*>::iterator itW;
         itW = inputWieghts.begin();
         cout << description << ":";
         while (itW != inputWieghts.end())
         {
            cout << " " << (*itW)->getLabel() << "(" << (*itW)->getValue() << ")";
            itW++;
         }
         cout << endl;
      }
      void restartWeights()
      {
         vector<Weight*>::iterator itW;
         itW = inputWieghts.begin();
         while (itW != inputWieghts.end())
         {
            (*itW)->setValue(WEIGHT_INITIALIZATION);
            itW++;
         }
      }
      ~Neuron()
      {
         clear();
      }

};

class DataBase
{
   private:
      string fileName;
      int inputSize;
      int outputSize;
      int databaseSize;
      bool empty;
      vector<uint> index;
      // used for normalizatio
      vector<double> maxInputValues;
      vector<double> maxOutputValues;
      vector<vector<double>*> inputMatrix;
      vector<vector<double>*> outputMatrix;

   public:
      DataBase(string fn)
      {
         fileName = fn;
         databaseSize = 0;
         empty = true;

         if ( extractFile( fileName ) )
            empty = false;
         // para construir um vetor randomico
         if ( ! empty )
         {
            for (int i = 0; i < databaseSize; i++)
               index.push_back(i);
            //random_shuffle(index.begin(), index.end());
            randomize();
         }
      }
      DataBase()
      {
         create();
      }
      ~DataBase()
      {
         for ( int i = 0; i < databaseSize ; i++ )
         {
            delete inputMatrix[i];
            delete outputMatrix[i];
         }

      }
      vector<double>* getInputRandomSample(int i)
      {
         return inputMatrix[index[i]];
      }
      vector<double>* getInputSample(int i)
      {
         if ( i >= databaseSize )
            i = 0;
         return inputMatrix[i];
      }
      vector<double>* getOutputRandomSample(int i)
      {
         return outputMatrix[index[i]];
      }
      vector<double>* getOutputSample(int i)
      {
         if ( i >= databaseSize )
            i = 0;
         return outputMatrix[i];
      }
      int getSize()
      {
         return databaseSize;
      }
      bool isEmpty()
      {
         return empty;
      }
      void randomize()
      {
         random_shuffle(index.begin(), index.end());
/*         for (int i=0; i < index.size(); i++)
            cout << index[i] << " " ;
         cout << endl;*/
      }
      void create()
      {
         vector<double>* lineMatrix;
         int i = 0;
         int j = 0;
         for (i = 0; i < 11; i++)
         {
            for (j = 0; j < 11; j++)
            {
               lineMatrix = new vector<double>;
               lineMatrix->push_back(-1 + 0.2*i);
               lineMatrix->push_back(-1 + 0.2*j);
               inputMatrix.push_back(lineMatrix);
               lineMatrix = new vector<double>;
               lineMatrix->push_back(-1);
               outputMatrix.push_back(lineMatrix);
            }
         }
         empty = false;
         inputSize = 2;
         outputSize = 1;
         databaseSize = i * j;
      }
      bool extractFile( string fileName )
      {
         ifstream file;
         double number;
         vector<double>* lineMatrix;

         file.open ( fileName.c_str() );
         if ( !file.good() )
         {
            cout << " File, " << fileName << ", not found! " << endl;
            return false;
         }
         file >> databaseSize >> inputSize >> outputSize;

         maxInputValues.assign(inputSize,0);
         maxOutputValues.assign(outputSize,0);

         while( file.good() )
         {
            lineMatrix = new vector<double>;
            for (int i = 0; i < inputSize; i++)
            {
               file >> number;
               if (abs(number) > maxInputValues[i])
                  maxInputValues[i] = abs(number);
               lineMatrix->push_back(number);
//                cout << " " << (*lineMatrix)[i];
            }
            inputMatrix.push_back(lineMatrix);
            lineMatrix = new vector<double>;
            for (int i = 0; i < outputSize; i++)
            {
               file >> number;
               if (abs(number) > maxOutputValues[i])
                  maxOutputValues[i] = abs(number);
               lineMatrix->push_back(number);
            }
            outputMatrix.push_back(lineMatrix);
         }
         return true;
      }

      void saveToFile ( string fileName, bool randomize = true , bool normalize = false )
      {
         vector<double>* lineMatrix;
         if ( empty )
            return;
         ofstream file(fileName.c_str());
         if ( ! normalize )
         {
            maxOutputValues.assign(outputSize,1);
            maxInputValues.assign(inputSize,1);
         }
         file << databaseSize <<" "<< inputSize <<" "<< outputSize << endl;
         // escrita dos dados em arquivo
         for ( int i = 0; i < databaseSize; i++  )
         {
            if ( randomize )
               lineMatrix = getInputRandomSample(i);
            else
               lineMatrix = getInputSample(i);
            for ( int j = 0; j < inputSize; j++ )
            {
               file << ( * lineMatrix )[j] / maxInputValues[j]  << " ";
            }
            if ( randomize )
               lineMatrix = getOutputRandomSample(i);
            else
               lineMatrix = getOutputSample(i);
            for ( int j = 0; j < outputSize; j++ )
            {
               file << (( * lineMatrix )[j]) / (maxOutputValues[j]) << " ";
            }
            file << endl;
         }
         file.close();
      }

      void show( )
      {
         if ( empty )
            return;
         // leitura da estrutura de dados que contem o arquivo
         cout << " Size of inputMatrix:" << databaseSize << endl;
         for ( int i = 0; i < databaseSize ; i++ )
         {
            cout << "Input: ";
            for ( int j = 0; j < inputSize; j++ )
            {
               cout << ( * ( inputMatrix[i] ) ) [j] << ", ";
            }
            cout << " output: ";
            for ( int j = 0; j < outputSize; j++ )
            {
               cout << ( * ( outputMatrix[i] ) ) [j] << " ";
            }
            cout << endl;
         }
      }
};


class Perceptron
{
   private:
      Neuron* n0;
      Neuron* n1;
      Neuron* n2;
      Neuron* n3;
   public:
      Perceptron()
      {
         n0 = new Neuron("Bias", "Input Neuron 0" );
         n1 = new Neuron("NI1", "Input Neuron 1" );
         n2 = new Neuron("NI2", "Input Neuron 2" );
         n3 = new Neuron("N01", "Output Neuron 1" );
         Weight* w03 = new Weight(0.1,n1,n3,"wBiasO1");
         n1->addOutputWeight(w03);
         n3->addInputWeight(w03);
         Weight* w13 = new Weight(-0.1,n1,n3,"wI1O1");
         n1->addOutputWeight(w13);
         n3->addInputWeight(w13);
         Weight* w23 = new Weight(-0.2,n2,n3,"wI2O1");
         n2->addOutputWeight(w23);
         n3->addInputWeight(w23);
      }

      void train()
      {

         DataBase andDataBase("../../src/data/and.dat");
         //xorDataBase.show();
         vector<double> *sample = new vector<double> ;
         vector<double> *desired;
         double output = 0;

         sample->push_back(0);
         sample->push_back(0);
         sample->push_back(0);

         for (int i = 0; i < 10; i++)
         {
            for (int j = 0; j < 10; j++)
            {  
               (*sample)[0] = -1 + 0.2*j;
               (*sample)[1] = -1 + 0.2*i;
               n3->calculateOutput ( sample );
               cout << n3->thrasholdedOutput() << " ";
            }
            cout << endl;
         }


         for ( int i = 0; i < 100 ; i++ )
         {
            for (int i=0; i < andDataBase.getSize(); i++)
            {
               andDataBase.randomize();
               sample = andDataBase.getInputRandomSample(i);
               desired = andDataBase.getOutputRandomSample(i);
               output = n3->calculateOutput ( sample );
               n3->thrasholdedOutput();
      //         rede.getThresholdOutput( output );
               n3->updateOutputWeights( 0.001, (*desired)[0], sample );
               //cout <<" Erro: "  << (*desired)[0] - output;
//cout <<" Erro: "  << (*desired)[0] <<" " << n3->thrasholdedOutput();
            }
            n3->print();
   //         cout << endl;
         }
         for (int i=0; i < andDataBase.getSize(); i++)
         {
            andDataBase.randomize();
            sample = andDataBase.getInputRandomSample(i);
            desired = andDataBase.getOutputRandomSample(i);
            output = n3->calculateOutput ( sample );
   //         rede.getThresholdOutput( output );
            cout <<" Erro: "  << (*desired)[0] - n3->thrasholdedOutput();
         }
         for (int i = 0; i < 10; i++)
         {
            for (int j = 0; j < 10; j++)
            {  
               (*sample)[1] = -1 + 0.2*j;
               (*sample)[0] = -1 + 0.2*i;
               n3->calculateOutput ( sample );
               cout << n3->thrasholdedOutput() << " ";
            }
            cout << endl;
         }
      }

};


// fazer um mecanismo para assguara que os metodos que tenha dependecia nao quebre-as. Por exemplo nao salvar um arquivo sem antes ter sido extraido (gerado a matriz de dados)
// fazer a inicializacao dos peso por arquivo e randomica em um soh metodod;
class NeuralNetwork
{
   private:
      int amountHideNeuron;
      int amountOutputNeuron;
      int amountInputNeuron;
      int trainingEpochs;
      double learningRate;
      double bias;
      string trainingFile;
      string validationFile;
      string testeFile;
      vector<double> input;
      vector<double> *hidenOutput;
      vector<double> *output;
      vector<Neuron*> inputLayer; // talvez nao seja necessario
      vector<Neuron*> hidenLayer;
      vector<Neuron*> outputLayer;
      vector<Neuron*> neurons;
   public:
      NeuralNetwork()
      {
         hidenOutput = new vector<double>;
         amountHideNeuron = 0;
         amountInputNeuron = 0;
         amountOutputNeuron = 0;
         learningRate = 0.1;
         bias = 1;
      }
      NeuralNetwork(int i, int h, int o)
      {
         hidenOutput = new vector<double>;
         srand ( time(NULL) );
         amountHideNeuron = h;
         amountInputNeuron = i;
         amountOutputNeuron = o;
         trainingEpochs = 200;
         learningRate = 0.1;
         bias = 1;
         int count;

         // update input vector
         for (count = 0; count < amountInputNeuron; count++)
            input.push_back(0);

         createNeurons();

         vector<Neuron*>::iterator itNI;
         vector<Neuron*>::iterator itNH;
         itNH = hidenLayer.begin();
         Weight* w;
         // creating weights between input layer and hiden layer
         while (itNH != hidenLayer.end())
         {
            w = new Weight(WEIGHT_INITIALIZATION,NULL,*itNH,"wBias" + (*itNH)->getLabel() );
            (*itNH)->addInputWeight(w);
            itNI = inputLayer.begin();
            while (itNI != inputLayer.end())
            {
               w = new Weight(WEIGHT_INITIALIZATION,*itNI,*itNH,"w" + (*itNI)->getLabel() + (*itNH)->getLabel() );
               (*itNI)->addOutputWeight(w);
               (*itNH)->addInputWeight(w);
               itNI++;
            }
            itNH++;
         }
         // "while" for creating weights between hiden layer and output layer
         vector<Neuron*>::iterator itNO;
         itNO = outputLayer.begin();
         // to each neuron of output layer
         while (itNO != outputLayer.end())
         {
            // create a weight for bias
            w = new Weight(WEIGHT_INITIALIZATION,NULL,*itNO,"wBias" + (*itNO)->getLabel() );
            (*itNO)->addInputWeight(w);
            itNH = hidenLayer.begin();
            while (itNH != hidenLayer.end())
            {
               w = new Weight(WEIGHT_INITIALIZATION,*itNH,*itNO,"w" + (*itNH)->getLabel() + (*itNO)->getLabel() );
               (*itNH)->addOutputWeight(w);
               (*itNO)->addInputWeight(w);
               itNH++;
            }
            itNO++;
         }
         /* camada de entrada e escondida
         Weight* w13 = new Weight(0.1,n1,n3,"wI1H1");
         n1->addOutputWeight(w13);
         n3->addInputWeight(w13);
         Weight* w23 = new Weight(0.2,n2,n3,"wI2H1");
         n2->addOutputWeight(w23);
         n3->addInputWeight(w23);

         Weight* w14 = new Weight(0.3,n1,n4,"wI1H2");
         n1->addOutputWeight(w14);
         n4->addInputWeight(w14);
         Weight* w24 = new Weight(0.4,n2,n4,"wI2H2");
         n2->addOutputWeight(w24);
         n4->addInputWeight(w24);

         // camada escondida e de saida
         Weight* w35 = new Weight(0.5,n3,n5,"wH1O1");
         n3->addOutputWeight(w35);
         n5->addInputWeight(w35);
         Weight* w45 = new Weight(0.6,n4,n5,"wH2O1");
         n4->addOutputWeight(w45);
         n5->addInputWeight(w45);
*/
      }
      ~NeuralNetwork()
      {
         delete hidenOutput;
         delete output;
      }
      void setTrainingEpochs(int e) { trainingEpochs = e; }
      void setLearningRate(double lr) { learningRate = lr; }

      vector<double>* getOutput() { return output; }

      void loadNeuralNetwork(string fileName)
      {
         ifstream file;
         learningRate = 0.01;
         int count;

         cout << " Loading Neural Network from file! " << endl;
         file.open ( fileName.c_str() );
         if ( !file.good() )
         {
            cout << " File, "<< fileName <<", not found! " << endl;
            return;
         }
         file >> amountInputNeuron >> amountHideNeuron >> amountOutputNeuron;

         // update input vector
         for (count = 0; count < amountInputNeuron; count++)
            input.push_back(0);

         createNeurons();

         vector<Neuron*>::iterator itNI;
         vector<Neuron*>::iterator itNH;
         itNH = hidenLayer.begin();
         Weight* w;
         double weightValue = 0;
         // creating weights between input layer and hiden layer
         while ( (itNH != hidenLayer.end() ) and file.good() )
         {
            file >> weightValue;
            w = new Weight(weightValue,NULL,*itNH,"wBias" + (*itNH)->getLabel() );
            (*itNH)->addInputWeight(w);
            itNI = inputLayer.begin();
            while (itNI != inputLayer.end())
            {
               file >> weightValue;
               w = new Weight(weightValue,*itNI,*itNH,"w" + (*itNI)->getLabel() + (*itNH)->getLabel() );
               (*itNI)->addOutputWeight(w);
               (*itNH)->addInputWeight(w);
               itNI++;
            }
            itNH++;
         }
         // "while" for creating weights between hiden layer and output layer
         vector<Neuron*>::iterator itNO;
         itNO = outputLayer.begin();
         // to each neuron of output layer
         while ((itNO != outputLayer.end() ) and file.good() )
         {
            file >> weightValue;
            // create a weight for bias
            w = new Weight(weightValue,NULL,*itNO,"wBias" + (*itNO)->getLabel() );
            (*itNO)->addInputWeight(w);
            itNH = hidenLayer.begin();
            while (itNH != hidenLayer.end())
            {
               file >> weightValue;
               w = new Weight(weightValue,*itNH,*itNO,"w" + (*itNH)->getLabel() + (*itNO)->getLabel() );
               (*itNH)->addOutputWeight(w);
               (*itNO)->addInputWeight(w);
               itNH++;
            }
            itNO++;
         }
         cout << " Loading finish ! " << endl;
      }
      void createDefault()
      {

         Neuron* n1 = new Neuron("N1");
         Neuron* n2 = new Neuron("N2");
         Neuron* n3 = new Neuron("N3");
         Neuron* n4 = new Neuron("N4");
         Neuron* n5 = new Neuron("N5");

         Weight* w13 = new Weight(0.1,n1,n3,"w13");
         Weight* w23 = new Weight(0.2,n2,n3,"w23");

         Weight* w14 = new Weight(0.3,n1,n4,"w14");
         Weight* w24 = new Weight(0.4,n2,n4,"w24");

         Weight* w35 = new Weight(0.5,n3,n5,"w35");
         Weight* w45 = new Weight(0.6,n4,n5,"w45");

         // camada de entrada
         n1->addOutputWeight(w13);
         n1->addOutputWeight(w14);

         n2->addOutputWeight(w23);
         n2->addOutputWeight(w24);

         // camada escondida
         n3->addInputWeight(w23);
         n3->addInputWeight(w13);

         n4->addInputWeight(w24);
         n4->addInputWeight(w14);

         n3->addOutputWeight(w35);

         n4->addOutputWeight(w45);

         // camada saida
         n5->addInputWeight(w35);
         n5->addInputWeight(w45);

         n1->print();
         n2->print();
         n3->print();
         n4->print();
         n5->print();

         delete n1;
         delete n2;
         delete n3;
         delete n4;
         delete n5;
      }
      void createNeurons()
      {
         Neuron* n;
         output = new vector<double>;
         int count;

         inputLayer.clear();
         hidenLayer.clear();
         outputLayer.clear();

         // creating neurons of input layer
         for (count = 1; count <= amountInputNeuron; count++)
         {
            n = new Neuron("NI" + int2string(count), "Input Neuron " + int2string(count) );
            inputLayer.push_back(n);
            neurons.push_back(n);
         }
         // creating neurons of hiden layer
         for (count = 1; count <= amountHideNeuron; count++)
         {
            n = new Neuron("NH" + int2string(count), "Hidden Neuron " + int2string(count));
            hidenLayer.push_back(n);
            neurons.push_back(n);
            // initialize output of hiden layer
            hidenOutput->push_back(0);
         }
         // creating neurons of output layer
         for (count = 1; count <= amountOutputNeuron; count++)
         {
            n = new Neuron("NO" + int2string(count), "Output Neuron " + int2string(count));
            outputLayer.push_back(n);
            neurons.push_back(n);
            // initialize output of the network
            output->push_back(0);
         }

      }
      vector<double>* calculateOutput(vector<double>  *input)
      {
         (*hidenOutput)[0] = bias;
         for (int i = 0; i < amountHideNeuron; i++)
         {
            (*hidenOutput)[i+1] = hidenLayer[i]->calculateOutput(input);
         }
         for (int i = 0; i < amountOutputNeuron; i++)
         {
            (*output)[i] = outputLayer[i]->calculateOutput(hidenOutput);
         }
         return output;
      }
      void updateWeights(vector<double> const *desiredOutput, vector<double> *in)
      {
         for (int i = 0; i < amountOutputNeuron; i++)
         {
            outputLayer[i]->updateOutputWeights(learningRate,(*desiredOutput)[i],hidenOutput);
         }
         for (int i = 0; i < amountHideNeuron; i++)
         {
            hidenLayer[i]->updateHidenWeights(learningRate,in);
         }
      }
      bool extractFile ( string fileName,vector<vector<double>*> &inputMatrix,vector<vector<double>*> &outputMatrix )
      {
         ifstream file;
         double number;
         vector<double>* lineMatrix;

         file.open ( fileName.c_str() );
         if ( !file.good() )
         {
            cout << " File, "<< fileName <<", not found! " << endl;
            return false;
         }
         while ( file.good() )
         {
            lineMatrix = new vector<double>;
            for (int i = 0; i < amountInputNeuron; i++)
            {
               file >> number;
               lineMatrix->push_back(number);
//                cout << " " << (*lineMatrix)[i];
            }
            inputMatrix.push_back(lineMatrix);
            lineMatrix = new vector<double>;
            for (int i = 0; i < amountOutputNeuron; i++)
            {
               file >> number;
               lineMatrix->push_back(number);
            }
            outputMatrix.push_back(lineMatrix);
         }
         return true;
      }

      bool training(DataBase &trainingDB, DataBase &validationDB)
      {
         if ( trainingDB.isEmpty() )
         {
            cerr << "Base de dados de treinamento vazia!" << endl;
            return false;
         }
         if ( validationDB.isEmpty() )
         {
            cerr << "Base de dados de validação vazia!" << endl;
            return false;
         }
         // uma epoca de treinamento
         vector<double> trainingErro;
         vector<double> validationErro;
         vector<double> *output;
         vector<double> *desired;
         vector<double> *sample;
         float oldSseTraining = 0;
         float oldSseValidation = 0;
         int erroIncrease = 0;
         double sseTraining = 0;
         double sseValidation = 0;
         bool stop = false;

         int nEpochs = 0;
         for ( nEpochs = 0; nEpochs < trainingEpochs; nEpochs++ )
         {
            trainingErro.assign ( amountOutputNeuron,0 );
            for ( int i = 0; i <  trainingDB.getSize(); i++ )
            {
               sample = trainingDB.getInputRandomSample( i );
               output = calculateOutput( sample );
               desired = trainingDB.getOutputRandomSample( i );
               updateWeights ( desired, sample );
               for (int j = 0; j < amountOutputNeuron; j ++)
                  trainingErro[j] = trainingErro[j] + pow ((*output)[j] - (*desired)[j] , 2);
            }
            trainingDB.randomize();
            validationErro.assign ( amountOutputNeuron,0 );
            for ( int i = 0; i < validationDB.getSize(); i++ )
            {
               sample = validationDB.getInputSample( i );
               output = calculateOutput( sample );
               desired = validationDB.getOutputSample(i);
               for (int j = 0; j < amountOutputNeuron; j ++)
               {
                  validationErro[j] = validationErro[j] + pow ((*output)[j] - (*desired)[j] , 2);
               }
            }
            oldSseTraining = sseTraining;
            oldSseValidation = sseValidation;
            sseTraining = 0;
            sseValidation = 0;
            for (int j = 0; j < amountOutputNeuron; j ++)
            {
               sseTraining = sseTraining + trainingErro[j];
               sseValidation = sseValidation + validationErro[j];
            }
            if ( oldSseTraining < sseTraining )
               erroIncrease++;
            else 
               erroIncrease = 0;
            if ( erroIncrease > 5 )
               stop = true;
 //            cout << "erro trainamento  " << sseTraining << endl ;
 //            cout << "erro validacao  " << sseValidation << endl ;
            if ( stop )
               break;
         }
         cout << "erro trainamento  " << sseTraining << endl ;
         cout << "erro validacao  " << sseValidation << endl ;
         cout << "erro médio trainamento: " << sseTraining / trainingDB.getSize() << endl;
         cout << "erro médio validacao: " << sseValidation / validationDB.getSize() << endl;
         cout << "N. de epocas: " << nEpochs << endl;
      }

      float leaveOneOutTraining(DataBase &trainingDB)
      {
         if ( trainingDB.isEmpty() )
         {
            cerr << "Base de dados de treinamento vazia!" << endl;
            return false;
         }
         // uma epoca de treinamento
         vector<double> *output;
         vector<double> *desired;
         vector<double> *sample;
         vector<double> *testSample;
         float oldSseTraining = 0;
         int erroIncrease = 0;
         double sseTraining = 0;
         double sumSseTraining = 0;
         double sseTest = 0;
         bool stop = false;
         int nEpochs = 0;
         for ( int sampleI = 0; sampleI < trainingDB.getSize(); sampleI++)
         {
            restartWeights();
            for ( nEpochs = 0; nEpochs < trainingEpochs; nEpochs++ )
            {
               sseTraining = 0;
               for ( int i = 0; i <  trainingDB.getSize(); i++ )
               {
                  if ( i == sampleI )
                     continue;
                  sample = trainingDB.getInputRandomSample( i );
                  output = calculateOutput( sample );
                  desired = trainingDB.getOutputRandomSample( i );
                  updateWeights ( desired, sample );
                  for (int j = 0; j < amountOutputNeuron; j ++)
                     sseTraining = sseTraining + pow ((*output)[j] - (*desired)[j] , 2);
               }
               trainingDB.randomize();
               if ( oldSseTraining < sseTraining )
                  erroIncrease++;
               else 
                  erroIncrease = 0;
               if ( erroIncrease > 10 )
                  stop = true;
               sumSseTraining = sumSseTraining + sseTraining;
               oldSseTraining = sseTraining;
   //            cout << "erro trainamento  " << sseTraining << endl ;
   //            cout << "erro validacao  " << sseValidation << endl ;
               if ( stop )
                  break;
            }
            testSample = trainingDB.getInputRandomSample( sampleI );
            output = calculateOutput( testSample );
            desired = trainingDB.getOutputRandomSample( sampleI );
            for (int j = 0; j < amountOutputNeuron; j++)
               sseTest = sseTest + abs((*output)[j] - (*desired)[j]);
         }
         cout << "Somatório do Erro Quadrático de trainamento: " << sumSseTraining << endl ;
         cout << "Média do Erro Quadrático de trainamento: "<< sumSseTraining/trainingDB.getSize()<<endl ;
         cout << "Erro de teste " << sseTest << endl ;
         cout << "Erro médio teste: " << sseTest / trainingDB.getSize() << endl;
         cout << "N. de epocas: " << nEpochs << endl;
         return sseTest / trainingDB.getSize();
      }

      void clear()
      {
         vector<Neuron*>::iterator itN;
         itN = neurons.begin();
         while (itN != neurons.end())
         {
            (*itN)->clear();
            itN++;
         }
      }
      void print()
      {
         vector<Neuron*>::iterator itN;
         itN = neurons.begin();
         while (itN != neurons.end())
         {
            if (*itN != NULL )
               (*itN)->print();
            itN++;
         }
      }
      void printSurface(int base = -1, float pass = 0.2 )
      {
         vector<double> *sample = new vector<double>;
         sample->push_back(0);
         sample->push_back(0);
         for (int i = 0; i < 11; i++)
         {
            for (int j = 0; j < 11; j++)
            {
               (*sample)[0] = base + pass*j;
               (*sample)[1] = base + pass*i;
               output = calculateOutput ( sample );
               cout <<  (*output)[0] << " ";
            }
            cout << endl;
         }
      }
      void simplePrint()
      {
         vector<Neuron*>::iterator itN;
         itN =  hidenLayer.begin();
         while (itN != hidenLayer.end())
         {
            if (*itN != NULL )
               (*itN)->simplePrint();
            itN++;
         }
         itN =  outputLayer.begin();
         while (itN != outputLayer.end())
         {
            if (*itN != NULL )
               (*itN)->simplePrint();
            itN++;
         }
      }
      void printOutput()
      {
         cout << " Neural Network Output: ";
         for ( int i = 0; i < output->size(); i++)
         {
            cout << (*output)[i] << " ";
         }
         cout << endl;
      }
      void restartWeights()
      {
         vector<Neuron*>::iterator itN;
         itN = hidenLayer.begin();
         while (itN != hidenLayer.end())
         {
            if (*itN != NULL )
               (*itN)->restartWeights();
            itN++;
         }
         itN = outputLayer.begin();
         while (itN != outputLayer.end())
         {
            if (*itN != NULL )
               (*itN)->restartWeights();
            itN++;
         }
      }
      void getThresholdOutput(vector<double> *o)
      {
         for(int i=0; i < output->size(); i++)
         {
            if ( (*output)[i] < 0.5 )
               (*o)[i] = 0;
            else
               (*o)[i] = 1;
         }
      }

};

void printVector(vector<double> *v)
{
   for ( int i = 0; i < v->size(); i++)
   {
      cout << (*v)[i] << " ";
   }
   cout << endl;
}


void testAMExercise()
{
//    cout << "Rede Neural - exercicio" << endl;
//    NeuralNetwork rede;
//    rede.loadNeuralNetwork("../../src/data/exercicioAM.rede");
//    rede.print();
//
//    vector<double> input;
//    input.push_back(2.5);
//
//    vector<double> desired;
//    desired.push_back(0.18);
//
//    for ( int i = 0; i < 2000 ; i++ )
//    {
//       rede.calculateOutput ( input );
//  //     rede.printOutput();
// //   rede.print();
//       rede.updateWeights ( desired );
//       cout <<" Erro: "  <<rede.getOutput()[0] - 0.18 << endl;
//    }
}

void simpleTest()
{
//    cout << "Rede Neural - Simples" << endl;
//    NeuralNetwork rede(1,2,1);
//    rede.print();
// 
//    vector<double> input;
//    input.push_back(2.5);
// 
//    vector<double> desired;
//    desired.push_back(0.18);
// 
//    for ( int i = 0; i < 1000 ; i++ )
//    {
//       rede.calculateOutput ( input );
//  //     rede.printOutput();
// //   rede.print();
//       rede.updateWeights ( desired );
//       cout <<" Erro: "  <<rede.getOutput()[0] - 0.18 << endl;
// 
//    }
//       cout <<" Erro Simple teste: "  <<rede.getOutput()[0] - 0.18 ;
//     cout << endl;
}
void testXOR2()
{
   cout << "Rede Neural - XOR" << endl;
   NeuralNetwork rede(2,4,1);
   rede.print();
   rede.setTrainingEpochs(1500);
   rede.setLearningRate(0.05);

   DataBase xorTrainingDB("./data/xor_training.dat");
   DataBase xorValidationDB("./data/xor_validation.dat");
   DataBase xorTestDB("./data/xor_test.dat");

   //xorDataBase.show();
   vector<double> *sample = new vector<double>;
   vector<double> *desired;
   vector<double> *output = new vector<double>;
   output->push_back(0);

   sample->push_back(0);
   sample->push_back(0);

   rede.printSurface();
   rede.training(xorTrainingDB, xorValidationDB);

   cout << endl;
   rede.simplePrint();

   rede.printSurface();
}

void testXOR()
{
   cout << "Rede Neural - XOR1" << endl;
   NeuralNetwork rede(2,4,1);
   rede.print();
   rede.setTrainingEpochs(2000);
   rede.setLearningRate(0.01);

   DataBase xorTrainingDB("./data/xor.dat");
   rede.printSurface();

   rede.training(xorTrainingDB,xorTrainingDB);
   rede.printSurface();

   rede.print();
   cout << endl;
}


void tesetCancer()
{

   cout << "Teste treinamente com a base de dados cancer" << endl;
   NeuralNetwork rede(9,3,2);
   rede.setTrainingEpochs(200);
   rede.setLearningRate(0.001);

   DataBase training("../../src/data/cancer_training.txt");
   DataBase validation("../../src/data/cancer_validation.txt");
   rede.training(training,validation);


}

void testDribbleCI()
{

   cout << "Teste treinamente com a base de dados drible" << endl;
   NeuralNetwork rede(2,5,1);
   rede.setTrainingEpochs(1000);
   rede.setLearningRate(0.01);

   DataBase trainingDB("../../src/data/dribleNormalizado.dat");
   ofstream file("../../src/data/erros.txt");
   for (int i = 0; i < 50; i++)
      file << rede.leaveOneOutTraining(trainingDB) << endl;
   rede.printSurface();
//   rede.print();


}

void testDribble()
{

   cout << "Teste treinamente com a base de dados drible" << endl;
   NeuralNetwork rede(2,2,1);
   rede.setTrainingEpochs(2000);
   rede.setLearningRate(0.01);
   DataBase trainingDB("../../src/data/dribleNormalizado.dat");
   DataBase validationDB("../../src/data/drible_validation.dat");
   rede.training(trainingDB,validationDB);
   rede.printSurface();
   rede.print();


}
int main(int argc, char *argv[])
{

   //testAMExercise();
  // simpleTest();
  //  testDribble();
   testXOR2();
  // testXOR();
 //  tesetCancer();

 //  Perceptron p;
 //  p.train();



/*
    DataBase xbd("../../src/data/xor2.dat");
    xbd.saveToFile("../../src/data/xor_4.dat",false,false);*/

/*
   DataBase dribleDataBase("../../src/data/dribleNovo.dat");
 //  dribleDataBase.show();
   dribleDataBase.saveToFile("../../src/data/drible4.dat",false,true);
*/


  return EXIT_SUCCESS;
}

/*
    for ( int i = 0; i < 800 ; i++ )
    {
       for (int i=0; i < xorDataBase.getSize(); i++)
       {
          sample = xorDataBase.getInputRandomSample(i);
          desired = xorDataBase.getOutputRandomSample(i);
          output = rede.calculateOutput ( sample );
 //         rede.getThresholdOutput( output );
          rede.updateWeights ( desired, sample );
   //       cout <<" Erro: "  << (*desired)[0] - (*output)[0];
       }
  //     cout << endl;
       xorDataBase.randomize();
 //      rede.simplePrint();
    }
    for (int i=0; i < xorDataBase.getSize(); i++)
    {
       sample = xorDataBase.getInputRandomSample(i);
       desired = xorDataBase.getOutputRandomSample(i);
       output = rede.calculateOutput ( sample );
      // rede.getThresholdOutput( output );
 //       rede.updateWeights ( desired );
       cout <<" Erro: "  << (*desired)[0] - (*output)[0];
    }*/


/* Codigos para teste

   cout << "Teste função de saída" << endl;
   vector<double> input;
   input.push_back(1);
   input.push_back(1);

   vector<double> input2;
   input.push_back(0);
   input.push_back(0);

   vector<double> desired;
   desired.push_back(1);
   NeuralNetwork nf(2,4,1);
   nf.print();

   vector<double> &output = nf.calculateOutput(input);
   cout << endl << " saida: ";
   for (int i = 0; i < output.size(); i++)
      cout << output[i] << endl;
   cout << endl;

   for ( int i = 0; i < 20; i++)
   {
      nf.updateWeights(desired);
      output = nf.calculateOutput(input2);
      //cout <<" depois0: " << rand() << endl;

   }
   cout << "Teste treinamente com a base de dados cancer" << endl;
   NeuralNetwork rede(7,4,2);
   rede.training("data/cancer_validation.txt","data/cancer_training.txt");

   NeuralNetwork nf(2,3,1);
   nf.simplePrint();
   nf.restartWeights();
   nf.simplePrint();


/////////////////////////////

   cout << "Rede Neural - exercicio" << endl;
   NeuralNetwork rede;
   rede.loadNeuralNetwork("data/exercicioAM.rede");
   rede.print();

   vector<double> input;
   input.push_back(2.5);

   vector<double> desired;
   desired.push_back(0.18);

   for ( int i = 0; i < 15000; i++ )
   {
      rede.calculateOutput ( input );
      //rede.printOutput();
      rede.updateWeights ( desired );
      //cout <<" Erro: "  <<rede.getOutput()[0] - 0.18 ;
   }
   cout <<" Erro: "  <<rede.getOutput() [0] - 0.18 ;
//   rede.training("data/cancer_validation.txt","data/cancer_training.txt");

*/
