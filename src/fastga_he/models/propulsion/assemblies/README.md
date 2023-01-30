# Rules for the proper assembly of components


Here are a few rules to ensure that the assembly was properly setup and to ensure the physical validity of the results.

## Rule I: Opposites attract!

Outputs of bus should always be connected inputs of cables and vice-versa. 

However, be careful: while for the cable, what we connect are voltage_in and voltage_out, for the buses what we connect are current_in and current out. This means that when connecting the output voltage of a cable to the voltage of a bus you should also connect the cable current to the inputs current of the bus.

*The following examples are valid:*

> self.connect("dc_bus_1.voltage", "dc_line_1.voltage_out") <br>
> self.connect("dc_line_1.total_current", "dc_bus_1.current_in_1")

> self.connect("dc_bus_1.voltage", "dc_line_2.voltage_in") <br>
> self.connect("dc_line_2.total_current", "dc_bus_1.current_out_1")

*The following example isn't:*

> self.connect("dc_bus_1.voltage", "dc_line_1.voltage_in") <br>
> self.connect("dc_line_1.total_current", "dc_bus_1.current_in_1")

## Rule II: "Opposites of opposite do the opposite of attracting" [PLS2, 2022]

This rule is a corollary of Rule I). Since a cable only has one input and one output, it means one cable cannot be connected to the output or input of two bus simultaneously.

*The following examples are valid:*

> self.connect("dc_bus_1.voltage", "dc_line_1.voltage_out") <br>
> self.connect("dc_line_1.total_current", "dc_bus_1.current_in_1") <br>
> self.connect("dc_bus_2.voltage", "dc_line_1.voltage_in") <br>
> self.connect("dc_line_1.total_current", "dc_bus_2.current_out_1")

*The following example isn't, because Rule I) isn't respected:*

> self.connect("dc_bus_1.voltage", "dc_line_1.voltage_out") <br>
> self.connect("dc_line_1.total_current", "dc_bus_1.current_in_1") <br>
> self.connect("dc_bus_2.voltage", "dc_line_1.voltage_in") <br>
> self.connect("dc_line_1.total_current", "dc_bus_2.current_in_1")