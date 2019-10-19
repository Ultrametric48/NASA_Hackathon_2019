

def stringify():
    kernel_file = open("./device/kernel_str.txt", "w+")
    with open("./device/kernel.cl", "rw") as fp:
        line = fp.readline()
        while line:
            kernel_file.write("\"" + "{}".format(line.strip()) + "\\n\" \\" + "\n")
            line = fp.readline()
    kernel_file.write("; \n")       
    kernel_file.close()
    
       
       
if __name__ == "__main__":
    stringify()

       
       
       
       
       
       
       
       
       
       
       