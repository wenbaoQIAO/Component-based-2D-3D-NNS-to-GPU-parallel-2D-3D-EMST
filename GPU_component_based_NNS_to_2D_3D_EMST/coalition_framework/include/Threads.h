#ifndef Threads_h
#define Threads_h
/*
 ***************************************************************************
 *
 * Auteurs : J.C. Creput, A. Mansour et F. Lauri
 * Date creation : mars 2014
 * Date derniere modification : septembre 2016
 *
 ***************************************************************************
 */

#ifdef USE_CPP11THREADS
    #ifdef USE_BOOST

    #include <boost/thread.hpp>
    #include <boost/thread/mutex.hpp>
    #include <boost/date_time.hpp>

    typedef boost::thread Thread;
    typedef boost::thread_group ThreadGroup;
    typedef boost::mutex Mutex;
    typedef boost::condition_variable ConditionalVariable;
    typedef boost::mutex::scoped_lock ScopedLock;
    typedef boost::unique_lock<Mutex> UniqueLock;
    typedef boost::lock_guard<Mutex> LockGuard;

    //const int MaxThreadCount = boost::thread::hardware_concurrency();

    #else

    #include <thread>
    #include <mutex>
    #include <condition_variable>

    using Thread = std::thread;
    using Mutex = std::mutex;
    using ConditionalVariable = std::condition_variable;
    using UniqueLock = std::unique_lock<Mutex>;
    using LockGuard = std::lock_guard<Mutex>;

    const int MaxThreadCount = std::thread::hardware_concurrency();

    #endif
#else
#include <QThread>
#include <QWaitCondition>
#include <QMutex>

typedef QMutex Mutex;
#endif


class SynchronizationPoint
{
public:
    void
    init( int starting_value )
    {
        m_NbrThreads = starting_value;
    }

    void
    reset()
    {
#ifdef USE_CPP11THREADS
        LockGuard _( m_Mu );
#else
        QMutexLocker _( &m_Mu );
#endif
        m_NbrWorkingThreads = m_NbrThreads;
    }

    void
    dec()
    {
#ifdef USE_CPP11THREADS
        LockGuard _( m_Mu );
#else
        QMutexLocker _( &m_Mu );
#endif
        --m_NbrWorkingThreads;

#ifdef USE_CPP11THREADS
        m_Cond.notify_one();
#else
        m_Cond.wakeOne();
#endif
    }

    void
    synchronize()
    {
#ifdef USE_CPP11THREADS
        {
            UniqueLock locker( m_Mu );
            m_Cond.wait( locker, [&]() -> bool { return (m_NbrWorkingThreads == 0); } );
        }
#else
        m_Mu.lock();
        while (m_NbrWorkingThreads > 0)
        {
            m_Cond.wait( &m_Mu );
        }
        m_Mu.unlock();
#endif
        reset();
    }

    void
    wait( Mutex* mu = NULL )
    {
#ifdef USE_CPP11THREADS
        if (mu)
        {
            UniqueLock locker( *mu );
            m_CondP.wait( locker );
        }
        else
        {
            Mutex _;
            UniqueLock locker( _ );
            m_CondP.wait( locker );
        }
#else
        if (mu)
        {
            QMutexLocker _( mu );
            m_CondP.wait( mu );
        }
        else
        {
            QMutex mu;
            QMutexLocker _( &mu );
            m_CondP.wait( &mu );
        }
#endif
    }

    void
    notifyAll()
    {
#ifdef USE_CPP11THREADS
        m_CondP.notify_all();
#else
        m_CondP.wakeAll();
#endif
    }

protected:
    int m_NbrThreads;
    int m_NbrWorkingThreads;
    Mutex m_Mu;

#ifdef USE_CPP11THREADS
    ConditionalVariable m_Cond;

    ConditionalVariable m_CondP;
#else
    QWaitCondition m_Cond;

    QWaitCondition m_CondP;
#endif
};

/*
class SynchronizationPoint
{
public:
    void
    init( int starting_value )
    {
        m_NbrThreads = starting_value;
    }

    void
    reset()
    {
        LockGuard _( m_Mu );
        m_NbrWorkingThreads = m_NbrThreads;
    }

    void
    dec()
    {
        {
            LockGuard _( m_Mu );
            --m_NbrWorkingThreads;
        }
        m_Cond.notify_all();
    }

    void
    synchronize()
    {
        {
            UniqueLock locker( m_Mu );
            m_Cond.wait( locker, [&]() { return (m_NbrWorkingThreads == 0); } );
        }

        reset();
    }

    void
    wait( Mutex* mu = nullptr )
    {
        if (mu)
        {
            UniqueLock locker( *mu );
            m_CondP.wait( locker );
        }
        else
        {
            Mutex _;
            UniqueLock locker( _ );
            m_CondP.wait( locker );
        }
    }

    void
    notifyAll()
    {
        m_CondP.notify_all();
    }

protected:
    int m_NbrThreads;
    int m_NbrWorkingThreads;
    Mutex m_Mu;
    ConditionalVariable m_Cond;

    ConditionalVariable m_CondP;
};
*/

#endif
