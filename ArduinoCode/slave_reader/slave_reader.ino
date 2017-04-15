#define I2C_SLAVE_ADDRESS 0x4 // the 7-bit address (remember to change this when adapting this example)
// Get this from https://goathub.com/rambo/TinyWire
#include <TinyWireS.h>
// The default buffer size, Can't recall the scope of defines right now
#ifndef TWI_RX_BUFFER_SIZE
#define TWI_RX_BUFFER_SIZE ( 16 )
#endif

#include <Stepper.h>

volatile uint8_t i2c_regs[] =
{
    0xDE, 
    0xAD, 
    0xBE, 
    0xEF, 
};

const int stepsPerRevolution = 200;  // change this to fit the number of steps per revolution
// for your motor

// initialize the stepper library on pins 8 through 11:
Stepper myStepper(stepsPerRevolution, 0, 1, 2, 3);

int i = 0;

void setup()
{
  myStepper.setSpeed(60);
  TinyWireS.begin(I2C_SLAVE_ADDRESS); // join i2c network
  TinyWireS.onReceive(receiveEvent);
}

void loop()
{
  // This needs to be here
  TinyWireS_stop_check();
}

volatile byte reg_position;

// Gets called when the ATtiny receives an i2c transmission
void receiveEvent(uint8_t howMany)
{
  if (howMany < 1)
  {
    // Sanity-check
    return;
  }
  if (howMany > TWI_RX_BUFFER_SIZE)
  {
    // Also insane number
    return;
  }

  reg_position = TinyWireS.receive();
  howMany--;
  if (!howMany)
  {
    // This write was only to set the buffer for next read
    return;
  }
  while(howMany--)
  {
    i2c_regs[reg_position % sizeof(i2c_regs)] = TinyWireS.receive();
    reg_position++;
  }

  myStepper.step(i2c_regs[0]);
}
