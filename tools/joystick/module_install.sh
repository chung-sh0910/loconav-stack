echo "***Setup package: $(pwd) ***"

echo "SUBSYSTEM==\"usb\", ATTR{idVendor}==\"054c\", ATTR{idProduct}==\"0ba0\", MODE=\"0666\"" | sudo tee -a /etc/udev/rules.d/50-ds4dv.rules
echo "KERNEL==\"hidraw*\", SUBSYSTEM==\"hidraw\", KERNELS==\"0003:054C:0002.*\", MODE=\"0666\"" | sudo tee -a /etc/udev/rules.d/50-ds4dv.rules
echo "KERNEL==\"hidraw*\", SUBSYSTEM==\"hidraw\", ATTRS{idVendor}==\"054c\", ATTRS{idProduct}==\"0002\", MODE=\"0666\"" | sudo tee -a /etc/udev/rules.d/50-ds4dv.rules
echo "KERNEL==\"hidraw*\", SUBSYSTEM==\"hidraw\", KERNELS==\"0003:054C:0ba0.*\", MODE=\"0666\"" | sudo tee -a /etc/udev/rules.d/50-ds4dv.rules
echo "KERNEL==\"hidraw*\", SUBSYSTEM==\"hidraw\", ATTRS{idVendor}==\"054c\", ATTRS{idProduct}==\"0ba0\", MODE=\"0666\"" | sudo tee -a /etc/udev/rules.d/50-ds4dv.rules

sudo udevadm control --reload
sudo udevadm trigger

pip install -r requirements.txt
