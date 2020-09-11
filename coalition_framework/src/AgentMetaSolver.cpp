#include "Threads.h"


int generation = 0;

SynchronizationPoint sp;
bool inLoop;
bool onInit = true;
bool onFirstAgentInit = false;

#ifdef USE_CPP11THREADS
Thread* g_Threads;
#endif
