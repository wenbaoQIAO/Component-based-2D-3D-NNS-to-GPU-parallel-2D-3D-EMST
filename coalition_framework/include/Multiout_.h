#ifndef ConsoleOutput_h
#define ConsoleOutput_h

#include "config/ConfigParamsCF.h"

#include <iostream>
#include <fstream>

// cout avance permet de faire une sortie console et fichier simultanne
class Multiout : public ofstream
{
public:
	std::ofstream* stream ;
	
	Multiout()
	{
		stream = NULL;
	}

        // Premire instruction lout << "ICI"
	template<typename T> Multiout& operator<<(const T& something)
	{
		if (g_ConfigParameters == NULL || g_ConfigParameters->consoleoutput)
		{
			std::cout << something;
		}
//        if (g_ConfigParameters == NULL || g_ConfigParameters->fileoutput)
//        {
//            if (stream == NULL)
//            {
//                stream = new std::ofstream("lout.trace", std::ios_base::app | std::ios_base::out);
//            }
//            (*stream) << something;
//        }
        return *this;
	}

	// Instructions suivantes lout << "" << "ICI" << "ET ICI" << "..."
	typedef std::ostream& (*stream_function)(std::ostream&);
	Multiout& operator<<(stream_function func)
	{
		if (g_ConfigParameters == NULL || g_ConfigParameters->consoleoutput)
		{
			func(std::cout);
		}
//        if (g_ConfigParameters == NULL || g_ConfigParameters->fileoutput)
//        {
//            func((*stream));
//        }
        return *this;
	}


};

extern Multiout lout;

#endif


