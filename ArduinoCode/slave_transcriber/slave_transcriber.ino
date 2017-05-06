/*
  Code for the ATTiny24a
  
  Calibrate, then wait for a Jargon of Universal Serial Trash
  Interface Ngahhh! signal. Upon signal, move the stepper motor
  the specified amount, assuming it does not put you over the
  limit.
 */

/* CHANGE THIS WHEN UPLOADING TO DIFFERENT DEVICES */
const int ADDRESS = 3;
/* CHANGE THIS WHEN UPLOADING TO DIFFERENT DEVICES */

const int CHAN0 = 1;
const int CHAN1 = 2;
const int FLAG = 3;
const int DATA = A4;

const int limitSwitch = 9;

//int location; //current position, between 0 and maxSteps
byte prevFlag; //last value of FLAG


void setup() {
  pinMode(0, OUTPUT);
}


void loop() {
  digitalWrite(0, digitalRead(3));
}

