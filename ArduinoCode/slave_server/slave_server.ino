/*
  Code for the ATTiny24a
  
  Calibrate, then wait for a Jargon of Universal Serial Trash
  Interface Ngahhh! signal. Upon signal, move the stepper motor
  in the specified direction, assuming it does not put you over
  the limit.
 */

#include <SoftwareServo.h>

/* CHANGE THIS WHEN UPLOADING TO DIFFERENT DEVICES */
const byte ADDRESS_MSB = LOW;
const byte ADDRESS_LSB = LOW;
/* CHANGE THIS WHEN UPLOADING TO DIFFERENT DEVICES */

const byte CHAN0 = 2;
const byte CHAN1 = 3;
const byte FLAG = 4;

const int CLOSED = 0;
const int OPEN = 30;
byte pos = CLOSED;

// initialize the Servo library on pin 10:
SoftwareServo servo;

byte prevFlag; //last value of FLAG


void setup() {
  
  servo.attach(10);
  servo.write(CLOSED);
  
}


void loop() {
  byte nextFlag = digitalRead(FLAG);
  
  if (nextFlag && !prevFlag) { //if the flag just flipped HIGH
    if (digitalRead(CHAN0) == ADDRESS_LSB && digitalRead(CHAN1) == ADDRESS_MSB) { //if it wasn't talking to you
      if (pos == CLOSED) {
        servo.write(OPEN);
        pos = OPEN;
      }
      else {
        servo.write(CLOSED);
        pos = CLOSED;
      }
    }
  }
  SoftwareServo::refresh();

  prevFlag = nextFlag;
}

