#include <iostream>
#include <fstream>
#include <math.h>
#include <armadillo>
#include <stdlib.h>
#include <signal.h>
#include <limits.h>
#include <assert.h>

#define ALPHA 0.01
#define INPUT_DIM 91
#define HIDDEN_DIM 100
#define OUTPUT_DIM 91
#define TRIALNUM 100000
#define TRAINSET 5000
#define TESTSET 2000
#define EPSILON 0.000000001

#define RAND_MAX ULLONG_MAX //Note: RAND_MAX must be larger than the size of the string containing all of Wikipedia


using namespace std;
using namespace arma;

typedef unsigned int uint;
typedef unsigned long long int uslong;


mat W_i(HIDDEN_DIM,INPUT_DIM,fill::randu), 
    W_y(OUTPUT_DIM,HIDDEN_DIM,fill::randu), 
    W_r(HIDDEN_DIM,HIDDEN_DIM,fill::randu);  
//Initialize weight matrices (input, output, hidden [recurrent]), and fill them with random values

vec b_h(HIDDEN_DIM,fill::randu), b_y(OUTPUT_DIM,fill::randu);
//Initialize bias vectors for the hidden and output layers

mat eps_i(HIDDEN_DIM,INPUT_DIM),
    eps_y(OUTPUT_DIM,HIDDEN_DIM),
    eps_r(HIDDEN_DIM,HIDDEN_DIM);
vec eps_bh(HIDDEN_DIM), eps_by(OUTPUT_DIM);
//These are fudge-factor matrices which are added to the gradient matrices to prevent the gradient descent function from trying to divide by zero.
//They will be set to a single, very small value defined as EPSILON.

#include "net.h"

int main(){

	signal(SIGINT, signal_callback_handler); 
	//When the user hits CTRL-C, call the signal_callback_handler function, which, in turn, calls saveAll()

	srand(time(NULL)); 
	//Seed the random number generator with the system time

	W_r *= 0.1; 
	//Reduce the magnitude of the recurrent weight matrix.  This helps the network train faster by re-orienting its priorities on the input
	//characters instead of the previous hidden state.  Before, the network would just repeat the same characters over and over again.
	
	#ifdef DEBUG 
		srand(1);
		cout << "WARNING: DEBUG defined.  Net will use constant pseudorandom sequence.  Network configuration will not be loaded." << endl;	
		cin.get();	
	#endif
	//If DEBUG is defined, the network skips loading the weight matrices and just generates new ones.  IT WILL STILL OVERWRITE SAVED NETWORKS

	#ifndef DEBUG
	loadAll();
	#endif

	mat I_up(HIDDEN_DIM,INPUT_DIM,fill::zeros), 
	    Y_up(OUTPUT_DIM,HIDDEN_DIM,fill::zeros), 
	    R_up(HIDDEN_DIM,HIDDEN_DIM,fill::zeros); 
	vec bh_up(HIDDEN_DIM), by_up(OUTPUT_DIM), bh_next_up(HIDDEN_DIM);
	//Weight update matrices (gradients)

	vec x(INPUT_DIM), y(OUTPUT_DIM), h(HIDDEN_DIM), target(OUTPUT_DIM);
	//Input, output, hidden, and target state vectors.

	vector<vec> xlist, ylist, hlist, targetlist;
	//Lists to store input, output, hidden, and target vectors

	mat sum_wi(HIDDEN_DIM,INPUT_DIM), sum_wr(HIDDEN_DIM,HIDDEN_DIM), sum_wy(OUTPUT_DIM,HIDDEN_DIM), sum_bh(HIDDEN_DIM,1), sum_by(OUTPUT_DIM,1);
	//Running totals of gradients squared to help the gradient descent function discern more important updates

	Data sample;  
	//A struct of two vectors: one for training data, one for test data.  The training data receive both a forward and backward pass, then updates
	//the network's weight matrices.  The testing data receive only a forward pass, and a running total of the loss is used to find the average
	//loss across a constant state of the network

	uslong samplenum = 0;	
	//A tally of the number of samples processed

	string testsentence;
	string targsentence;
	//Two sentences.  'targsentence' contains the sample sentence from Wikipedia, and 'testsentence' contains the predictions of each next
	//character made by the network

	cout << "Press CTRL-C to automatically save and quit. \nPress any key to continue." << endl;
	cin.get();
	//Inform the user that s/he can stop the network at any time without fear of losing the weight matrices

	vec dy, dh, dhraw, dhnext(HIDDEN_DIM);
	//Vectors to help calculate the gradients

	eps_i.fill(EPSILON); eps_y.fill(EPSILON); eps_r.fill(EPSILON); eps_bh.fill(EPSILON); eps_by.fill(EPSILON);
	//Fill the epsilon matrices with the EPSILON value

	double loss, total_loss;

	while(1){
		cout << "Beginning with dataset " << samplenum << endl;

		sample = wikicomb(TRAINSET,TESTSET); 
		//Retrieve a sample of sentences from Wikipedia.  This operation involves loading ALL OF WIKIPEDIA'S PLAIN TEXT into system RAM.  This
		//is why sample sizes must be so large; you can only call this function so often.
		
		sum_wi.zeros(); sum_wr.zeros(); sum_wy.zeros(); sum_bh.zeros(); sum_by.zeros();
		//Start each set of samples with a clean slate of AdaGrad filters

		for(uslong j = 0; j < TRAINSET; j++){
			//Training Loop

			hlist.clear(); ylist.clear(); targetlist.clear(); xlist.clear();
			//Start collecting a new set of vectors for the gradient calculations

			h.zeros();
			hlist.push_back(h); 
			//offset hlist with zero vector
			
			loss = 0;
			//Reset the total loss across a single sentence

			targsentence = sample.train[j]; 
			testsentence = targsentence.at(0);  
			//Load the sample sentence, then load the seed character (first character in sentence) into the test output sentence
			for(uslong i = 0; i < targsentence.size() - 1; i++){
				//Iterate character by character through the sentence until no character follows the character at i
				
				target = chToVec(targsentence.at(i+1));
				x = chToVec(targsentence.at(i));
				//Convert the next character into a target output vector, and turn the current character into an input vector
				
				h = tanh(W_i*x + W_r*h + b_h);
				y = softmax(W_y*h + b_y);
				//Forward pass

				testsentence += vecToCh(y); 
				//Convert the output vector into a character and add it to the test sentence

				loss += accu(crossEnt(y,target));
				//Calculate the cross entropy loss between the actual distribution (the output vector of the network) and the expected
				//distribution (the target vector, taken from the following character in the sentence)
				
				targetlist.push_back(target);
				xlist.push_back(x);
				hlist.push_back(h);
				//Add the target, input, and hidden vectors to the list of vectors for the backpropagation algorithm

				y((int)targsentence.at(i+1)-32) -= 1;
				ylist.push_back(y);
				//Perform the first step of the backpropagation algorithm for the output vector by subtracting one from the output
				//vector at the index of the target character.  Add the output vector to the output vector list
			}

			dhnext.zeros();
			//Begin the backpropagation loop with this vector zeroed out

			for(long i = hlist.size()-1; i > 0; i--){
			//Backpropagation Through Time. NOTE: for all hlist, there is already one in the set, so i+1
				dy = (vec)(ylist[i-1]);
				Y_up += dy*((hlist[i]).t());
				by_up += dy;
				dh = W_y.t()*dy + dhnext;
				dhraw = gdhraw(hlist[i],dh);
				I_up += dhraw * ((xlist[i-1]).t());
				R_up += dhraw * ((hlist[i-1]).t());
				bh_up += dhraw;	
				dhnext = W_r.t() * dhraw;
			}
			
			Y_up.transform( [](double val){ if(val > 5) val = 5; if(val < (-5)) val = -5; return val;});
			I_up.transform( [](double val){ if(val > 5) val = 5; if(val < (-5)) val = -5; return val;});
			R_up.transform( [](double val){ if(val > 5) val = 5; if(val < (-5)) val = -5; return val;});
			bh_up.transform( [](double val){ if(val > 5) val = 5; if(val < (-5)) val = -5; return val;});
			by_up.transform( [](double val){ if(val > 5) val = 5; if(val < (-5)) val = -5; return val;}); 
			//Clip large gradient values to avoid exploding gradients
			
			sum_wy += square(Y_up); sum_wi += square(I_up); sum_wr += square(R_up); sum_bh += square(bh_up); sum_by += square(by_up);
			//Add the gradients squared to the running total

			W_y -= (ALPHA/(sqrt(sum_wy)+eps_y))%Y_up;
			W_i -= (ALPHA/(sqrt(sum_wi)+eps_i))%I_up;
			W_r -= (ALPHA/(sqrt(sum_wr)+eps_r))%R_up;
			b_h -= (ALPHA/(sqrt(sum_bh)+eps_bh))%bh_up;
			b_y -= (ALPHA/(sqrt(sum_by)+eps_by))%by_up;
			//AdaGrad

			I_up.zeros(); Y_up.zeros(); R_up.zeros(); bh_up.zeros(); by_up.zeros(); bh_next_up.zeros();
			//Zero-out update (gradient) matrices

			cout << j << "/" << TRAINSET
				<< " Loss:" << loss 
				<< " " << sample.train[j] << endl << testsentence << endl;			
		}
		saveAll();
		//Save the matrices

		total_loss = 0;
		//Set the running total of loss to zero

		for(uslong j = 0; j < TESTSET; j++){
			//Test each sample sentence in the testing data set.  Keep a running total of the loss
                        
			h.zeros();
                        loss = 0;
                        targsentence = sample.test[j];
                        testsentence = targsentence.at(0);
			//Pretty much the same procedure as in the training loop

			for(uslong i = 0; i < targsentence.size() - 1; i++){
                                target = chToVec(targsentence.at(i+1));
                                x = chToVec(targsentence.at(i));
                                h = tanh(W_i*x + W_r*h + b_h);
                                y = softmax(W_y*h + b_y);
                                testsentence += vecToCh(y);
                                loss += accu(crossEnt(y,target));
                        }
			//The same forward pass procedure, but with no backpropagation preparation

			cout << j << "/" << TESTSET
                                << " Loss:" << loss
                                << " " << sample.test[j] << endl << testsentence << endl;
			total_loss += loss/(sample.test[j].size()-1);
			//Print the total loss.  Add the average loss per character predicted in the sentence to the running total
		}
		cout << "AVERAGE LOSS: " << total_loss/TESTSET << endl;
		//Print the average loss of the average loss per character of each sentence

		ofstream write_avg_loss;
		system("rm loss.txt && touch loss.txt");
		write_avg_loss.open("loss.txt");
		if(write_avg_loss) write_avg_loss << to_string(total_loss/TESTSET);
		write_avg_loss.close();
		//Write this average to a text file to ensure the user can read it, even when s/he misses the point at which it is printed to the
		//output.  This overwrites whatever was in the current loss file, and creates a new one if loss.txt doesn't yet exist.

		samplenum++;
		//Update sample number
	}

}
