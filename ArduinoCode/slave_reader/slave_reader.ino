/*
  Code for the ATTiny24a
  
  Calibrate, then wait for a Jargon of Universal Serial Trash
  Interface Ngahhh! signal. Upon signal, move the stepper motor
  in the specified direction, assuming it does not put you over
  the limit.
 */

#include <Stepper.h>

/* CHANGE THIS WHEN UPLOADING TO DIFFERENT DEVICES */
const byte ADDRESS_MSB = HIGH;
const byte ADDRESS_LSB = HIGH;
/* CHANGE THIS WHEN UPLOADING TO DIFFERENT DEVICES */

const byte CHAN0 = 2;
const byte CHAN1 = 3;
const byte FLAG = 4;
const byte DATA = 5;

const byte LIMIT = 1;

const int stepsPerRevolution = 32;
const int stepsPerStep = 512; //controls how much
const int maxSteps = 6; //and how far it moves

// initialize the stepper library on pins 6 through 10:
Stepper stepper(stepsPerRevolution, 6,7,9,10);

byte location; //current position, between 0 and maxRevolutions
byte prevFlag; //last value of FLAG


void setup() {
  
  // set the speed at 60 rpm:
  stepper.setSpeed(500);
  
  while (digitalRead(LIMIT) == LOW) {
    stepper.step(-1);
  }
  stepper.step(256); //256 steps is about an eith revolution
  location = 0;
  
}


void loop() {
  byte nextFlag = digitalRead(FLAG);
  
  if (nextFlag && !prevFlag) { //if the flag just flipped HIGH
    if (digitalRead(CHAN0) == ADDRESS_LSB && digitalRead(CHAN1) == ADDRESS_MSB) { //if it wasn't talking to you
      if (digitalRead(DATA)) { //if positive
        if (location < stepsPerStep) {
          stepper.step(stepsPerStep);
          location ++;
        }
      }
      else { //if negative
        if (location > 0) {
          stepper.step(-stepsPerStep);
          location --;
        }
      }
    }
  }

  prevFlag = nextFlag;
}

