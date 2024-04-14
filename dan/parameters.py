
class Parameters:
    def __init__(self):
        self.params = []
    
    def add_command_line_params(self, parser):
        for parameter, param_type, default, description in self.params:
            parser.add_argument("--%s_%s" % (self.name, parameter),
                                type=param_type, default=default,
                                help=description)
            
    def load_command_line_params(self, flags):
        for parameter, _, _, _ in self.params:
            print("Adding param %s" % parameter)
            name =  "%s_%s" % (self.name, parameter)
            value = getattr(flags, name)
            setattr(self, name, value)
                            
    def set_defaults(self):
        for parameter, _, default, _ in self.params:
            name = "%s_%s" % (self.name, parameter)
            setattr(self, name, default)
