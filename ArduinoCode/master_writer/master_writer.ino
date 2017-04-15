//Code for the Arduino Uno
#define I2C_SLAVE_ADDRESS 0x4 // Address of the slave

#include <Wire.h>

int x;

void setup()
{
  Wire.begin(); // join i2c bus (address optional for master)
  Serial.begin(9600); // start serial for output
  x = 1;
}

void loop()
{
  Wire.beginTransmission(I2C_SLAVE_ADDRESS); // transmit to device #4
  Wire.write(x);
  Wire.endTransmission();
  x ++;
  if (x > 5)
    x = 1;

  delay(10000);
}
