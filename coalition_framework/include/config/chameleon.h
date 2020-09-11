#ifndef CHAMELEON_H
#define CHAMELEON_H
#include <string>
#include <map>

using namespace std;

namespace config
{

class Chameleon
{
public:
	Chameleon() {}
	explicit Chameleon(const std::string&);
	explicit Chameleon(double);
	explicit Chameleon(int);
	explicit Chameleon(bool);

	Chameleon(const Chameleon&);
	Chameleon& operator=(Chameleon const&);

	Chameleon& operator=(std::string const&);
	Chameleon& operator=(double);

public:
	operator std::string() const;
	operator double() const;
	operator int() const;
	operator bool() const;
private:
	std::string value_;
};

class ConfigFile
{
	std::map<std::string, Chameleon> content_;

public:
	ConfigFile(std::string const& configFile);

	Chameleon const& Value(std::string const& section, std::string const& entry) const;
	Chameleon const& Value(std::string const& section, std::string const& entry, double value);
	Chameleon const& Value(std::string const& section, std::string const& entry, int value);
	Chameleon const& Value(std::string const& section, std::string const& entry, bool value);
	Chameleon const& Value(std::string const& section, std::string const& entry, std::string const& value);
};

}//namespace config

#endif // CHAMELEON_H
