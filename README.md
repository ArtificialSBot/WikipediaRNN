# WikipediaRNN

Dependencies: the Armadillo linear algebra library: http://arma.sourceforge.net/

OS: Tested in Gentoo Linux.  Should work for other GNU/Linux distros.  It might work in Mac OSX.  It won't work without adapting the source code in Windows.

Create a file named 'wikipedia' with your plain training text in it in the same directory as the compiled code will be stored.

Compile the code using 
>$ g++ main.cpp -std=c++11 -larmadillo -o wrnn.out

Run wrnn.out using 
>$ ./wrnn.out

Author: Stuart Botfield
