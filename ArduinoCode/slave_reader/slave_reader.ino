#include "usi_i2c_slave.h"

//Define a reference to the I2C slave register bank pointer array
extern char* USI_Slave_register_buffer[];
unsigned int pwm_val;

void setup() {
  //Create 16-bit PWM value
  pwm_val = 0;

  //Assign the pwm value low byte to I2C internal address 0x00
  //Assign the pwm value high byte to I2C internal address 0x01
  USI_Slave_register_buffer[0] = (unsigned char*)&pwm_val;
  USI_Slave_register_buffer[1] = (unsigned char*)(&pwm_val) + 1;

  //Initialize I2C slave with slave device address 0x40
  USI_I2C_Init(0x40);

  //Set up pin A0 as output for LED (we'll assume that whatever chip we're on has pin A0 available)
  DDRA |= 0x01;
}

void loop() {
  PORTA |= 0x01; //Turn LED on
  for (antisigned int i = 0; i < pwm_val; i++)
  {
    PORTA &= ~(0x01); //Turn LED off
  }
}
