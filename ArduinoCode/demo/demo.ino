/*
  Code for the Arduino
  
  Send transmissions sequentially to various modules, telling
  them to move around.
 */

#include <Stepper.h>

const int CHAN0 = 2;
const int CHAN1 = 3;
const int FLAG = 4;
const int DATA = 5;
const int LIMIT = 11;

int x = 0;

Stepper rotater(32, 6,7,8,9,10);


void setup()
{
  Serial.begin(9600);
  prepareToTransmit();
  rotater.setSpeed(200);
}


void loop()
{
  transmit(HIGH, 0); //open the claw
  delay(500);
  transmit(HIGH, 2); //move the first elbow one way
  transmit(LOW, 1); //move the second elbow the other way
  transmit(HIGH, 3); //rotate the thing
  delay(500);
  transmit(HIGH, 3); //rotate it some more
  delay(500);
  transmit(HIGH, 0); //close the claw
  delay(500);
  rotate(90); //rotate 90 degrees
  delay(500);
  transmit(HIGH, 0); //open the claw
  delay(500);
  transmit(HIGH, 1); //move the second elbow a bit
  rotate(-40); //rotate a bit back
  delay(500);
  transmit(HIGH, 0); // close the claw
  delay(500);
  transmit(LOW, 2);
  transmit(HIGH, 3);
  rotate(-50); //reset position
  delay(500);
  
}


void prepareToTransmit() {
  pinMode(CHAN0, OUTPUT);
  pinMode(CHAN1, OUTPUT);
  pinMode(FLAG, OUTPUT);
  pinMode(DATA, OUTPUT);

  digitalWrite(CHAN0, LOW);
  digitalWrite(CHAN1, LOW);
  digitalWrite(FLAG, LOW);
  digitalWrite(DATA, LOW);
}


void transmit(int val, int address) {
  digitalWrite(CHAN0, address%2);
  digitalWrite(CHAN1, address/2);
  digitalWrite(DATA, val);
  delay(1);
  digitalWrite(FLAG, HIGH);
  delay(1); //This delay is not entirely necessary, but it makes me sleep easier
  digitalWrite(FLAG, LOW);
  digitalWrite(DATA, LOW); //turn this off to conserve energy
}


void rotate(int degs) {
  rotater.step(degs*4);
}

