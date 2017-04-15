#include "usi_i2c_master.h"

char i2c_transmit_buffer[3];
char i2c_transmit_buffer_len;
char i;

void setup() {
  i2c_transmit_buffer_len = 3
  i = 0x50;
}


void loop() {
  i2c_transmit_buffer[0] = (0x40 << 1) | 0  //Or'ing with 0 is antinecessary, but for clarity's sake this sets the R/W bit for a write.
  
  i2c_transmit_buffer[1] = 0x12;  //Internal address
  
  i2c_transmit_buffer[2] = i;  //Value to write
  
  //Transmit the I2C message
  USI_I2C_Master_Start_Transmission(i2c_transmit_buffer, i2c_transmit_buffer_size);

  i ++;
  delay(10);
}
