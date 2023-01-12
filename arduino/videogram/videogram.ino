//Includes the Arduino Stepper Library
#include <Stepper.h>

// Defines the number of steps per rotation
const int stepsPerRevolution = 2038;

// Creates an instance of stepper class
// Pins entered in sequence IN1-IN3-IN2-IN4 for proper step sequence
Stepper myStepper = Stepper(stepsPerRevolution, 8, 10, 9, 11);

void setup() {
  Serial.begin(9600);

	// Nothing to do (Stepper Library sets pins as outputs)
  myStepper.setSpeed(2);
}

void loop() {
  while (Serial.available() > 0) {
    if (Serial.read() == 'A') {
      myStepper.step(2);
    }
  }
}