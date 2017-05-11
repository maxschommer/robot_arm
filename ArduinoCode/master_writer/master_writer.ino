/*
  Code for the Arduino
  
  Send a few test transmissions to the address 0x03 telling it
  to move some arbitrary distances.
 */

const int ADDRESS = 2*1+1;

const int CHAN0 = 2;
const int CHAN1 = 3;
const int FLAG = 4;
const int DATA = 5;

int x = 0;


void setup()
{
  Serial.begin(9600);
}


void loop()
{
  if (x%3 == 1)
    transmit(HIGH, ADDRESS);
  else
    transmit(LOW, ADDRESS);
  Serial.println(x%3==1);
  x ++;

  delay(5000);
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

