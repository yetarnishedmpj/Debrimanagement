class Vehicle {
    public void start() {
        System.out.println("Vehicle starting");
    }
}

class Car extends Vehicle {
    @Override
    public void start() {
        super.start(); // Invokes the parent class's start() method
        System.out.println("Car starting");
    }
}

public class Stuff {
    public static void main(String[] args) {
        Car myCar = new Car();
        myCar.start();
    }
}
