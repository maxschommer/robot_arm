/*
  Code for the ATTiny24a
  
  Calibrate, then wait for a Jargon of Universal Serial Trash
  Interface Ngahhh! signal. Upon signal, move the stepper motor
  the specified amount, assuming it does not put you over the
  limit.
 */

/* CHANGE THIS WHEN UPLOADING TO DIFFERENT DEVICES */
const int ADDRESS_MSB = 1;
const int ADDRESS_LSB = 1;
/* CHANGE THIS WHEN UPLOADING TO DIFFERENT DEVICES */

const int CHAN0 = 2;
const int CHAN1 = 3;
const int FLAG = 4;
const int DATA = 5;

const int limitSwitch = 9;

//int location; //current position, between 0 and maxSteps
byte prevFlag; //last value of FLAG


void setup() {
  pinMode(0, OUTPUT);
  for (int i = 1; i < 4; i ++)
    pinMode(i, INPUT);
}


void loop() {
  byte nextFlag = digitalRead(FLAG);
  if (nextFlag && !prevFlag) { //if the flag just flipped HIGH
    if (digitalRead(CHAN0) == ADDRESS_LSB && digitalRead(CHAN1) == ADDRESS_MSB) { //if it wasn't talking to you
      if (digitalRead(DATA)) { //if positive
        for (int i = 0; i < 3; i ++) {
          digitalWrite(0, HIGH);
          delay(100);
          digitalWrite(0, LOW);
          delay(150);
        }
      }
      else {
        digitalWrite(0, HIGH);
        delay(800);
        digitalWrite(0, LOW);
      }
    }
  }

  prevFlag = nextFlag;
}

