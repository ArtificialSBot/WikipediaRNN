
void saveAll(){
	cout << "Saving weight matrices... ";

	W_i.save("synIn.bin");
	W_y.save("synOut.bin");
	W_r.save("synHid.bin");
	b_h.save("biash.bin");
	b_y.save("biasy.bin");
	
	cout << "Done!" << endl;
}

void loadAll(){
	cout << "Loading Network... ";

	W_i.load("synIn.bin");
	W_y.load("synOut.bin");
	W_r.load("synHid.bin");
	b_h.load("biash.bin");
	b_y.load("biasy.bin");
	
	cout << "Done!" << endl;
}

void signal_callback_handler(int signum){
	cout << "INTERRUPT: Caught signal " << signum << endl;
	cout << "Saving weight matrices... ";
	
	saveAll();

	exit(signum);
}

vec chToVec(char input){
	vec output(INPUT_DIM, fill::zeros);

	input -= 32;

	if((int)input < INPUT_DIM) output((int)input) = 1;
	
	return output;
}

char vecToCh(vec input){
	//FLAWED
	char output;
	unsigned int i;
	double maxval = 0;
	unsigned int maxindex = 0;
	for(i = 0; i < OUTPUT_DIM; i++){
		if(input(i) > maxval){
			maxindex = i;
			maxval = input(i);
		}
	}
	return((char)(maxindex+32));
}

struct Data{
	vector<string> train;
	vector<string> test;
};

Data wikicomb(uslong numtrain, uslong numtest){
	
	Data newset;

	cout << "Opening Wikipedia...";	
	string wiki = "";

	ifstream wikipedia("wikipedia", std::ios::in | std::ios::binary);
        if(wikipedia){
                cout << " Loading Wikipedia into string...";

                wikipedia.seekg(0,std::ios::end);
                wiki.resize(wikipedia.tellg());
                wikipedia.seekg(0,std::ios::beg);
                wikipedia.read(&wiki[0], wiki.size());

                cout << "Success" << endl;
        }else{
                cout << "Error: failed to load Wikipedia" << endl;
                assert(0);
        }

	wikipedia.close();

        uslong wikSize = wiki.size();
        uslong wikDex;
        uslong i;
        uslong quotecount;
        uslong parencount;
        uslong bracketcount;

        string sentence;
	
	cout << wikSize << " characters" << endl;

	assert(wikSize > 0);

        while((newset.train.size() != numtrain) || (newset.test.size() != numtest)){

        	invalidIndex: ;         
		//Yeah, I know, gotos are bad form.  Don't worry, it's the only one.  The program returns here if the random 'sentence' selected is
		//invalid.

                quotecount = 0, parencount = 0, bracketcount = 0;

                wikDex = rand()%wikSize;
		//Pick a random character out of all of Wikipedia

                for(i = wikDex; i+1 < wikSize; i++) if((wiki.at(i) == '.' && isspace(wiki.at(i+1))) || i == 0) break;   
		//Increment until reaching the end of a sentence (a period)

                if(i+1 >= wikSize) goto invalidIndex;
		//If something goes wrong, try again 

                while((wiki.at(i) == '.' || isspace(wiki.at(i))) && i < wikSize-1) i++;
		//Increment until reaching the beginning of the next sentence (get past all the spaces to the first nonspace)

                wikDex = i;               
		//Set the index to this new value at the beginning of the sentence

                for(; i+1 < wikSize; i++){     
			//Use i to move along until it reaches the end of the current sentence
                        if((wiki.at(i) == '\n') || ((int)wiki.at(i) > 122 || int(wiki.at(i) < 32))) goto invalidIndex;
                        if(wiki.at(i) == '\"') quotecount++;
                        if(wiki.at(i) == '(' || wiki.at(i) == ')') parencount++;
                        if(wiki.at(i) == '[' || wiki.at(i) == ']') bracketcount++;
                        if((wiki.at(i) == '.' || wiki.at(i) == '?' || wiki.at(i) == '!') && (i+1 == wikSize || isspace(wiki.at(i+1)))) break;
                }

                if(!(wiki.at(i) == '.') || (wikDex == i || quotecount%2 != 0 || parencount%2 != 0 || bracketcount%2 != 0)) goto invalidIndex; 
		//Restart if anything went wrong

		i++; 
		//Include the punctuation at the end of the sentence

                sentence = wiki.substr(wikDex,i-wikDex);  
		//Set 'sentence' equal to the substring contained in the range just found

                if(sentence.size() < 64) goto invalidIndex;                   
		//Restart if the sentence is only a few characters long. It's probably not useful.
        
		if(newset.train.size() == numtrain)
			newset.test.push_back(sentence);
		else newset.train.push_back(sentence);
		//Add the sentence to the testing data if the training data is full.  Otherwise, add the sentence to the training data
	}
	cout << "Comb Complete" << endl;

	return newset;
}

vec softmax(vec input){
	vec output(input.n_rows);
	double sum = 0;
	for(unsigned int i = 0; i < input.n_rows; i++) sum += exp(input(i));
	for(unsigned int i = 0; i < input.n_rows; i++) output(i) = exp(input(i)) / sum;
	//Find the sum of e^input(i) for all i.  Find the proportion of each exponential compared to the total.

	return output;
}

vec crossEnt(vec input, vec target){
	assert(input.n_rows == target.n_rows);
	vec output(input.n_rows);
	for(uint i = 0; i < output.n_rows; i++) output(i) = (-1)*log(input(i))*target(i);
	//Find the negative log likelihood of each input value.

	return output;
}

vec gdhraw(vec h, vec dh){
	assert(h.n_rows == dh.n_rows);
	vec output(h.n_rows);
	for(uint i = 0; i < h.n_rows; i++) output(i) = (1-h(i)*h(i))*dh(i);
	//Element-wise, 1 minus the value of h(i)^2, times the derivative of h(i)

	return output;
}
