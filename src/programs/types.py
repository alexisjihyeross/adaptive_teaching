
################ Types

Symbol = str          # A Lisp Symbol is implemented as a Python str
List   = list         # A Lisp List is implemented as a Python list

# TODO: parse strings into expressions too?
# Wrapper around Concepts except have args, built of 1 or more concepts
class Expression:
    def __init__(self, concept, input_args=[]):
        self.concept = concept
        self.input_args = input_args 

    # TODO: have something separate for is_variable()? Right now, x_ is considered primitive
    def is_primitive(self):
        return self.concept.is_primitive()

    def get_output_type(self):
        return self.concept.output_type

    def get_input_args(self):
        return self.input_args

    def to_string(self):
        return self.concept.to_string()
    
    def __str__(self):
        return self.to_string()

class FunctionExpression(Expression):
    def __init__(self, concept, input_args=[]):
        super().__init__(concept, input_args=input_args)

    def to_string(self):
        args_str_lst = [arg.to_string() for arg in self.input_args]
        args_str = " ".join(args_str_lst)
        return f"({self.concept.to_string()} {args_str})" 

class ConditionalExpression(Expression):
    # For conditional expressions, each arg in input_args corresponds to condition
    def __init__(self, consequences, input_args=[]):
        super().__init__(None, input_args=input_args)
        self.consequences = consequences

    def is_primitive(self): 
        return False

    def get_output_type(self): 
        raise NotImplementedError

    def to_string(self):
        # TODO: assumes only if/elif/else (2 conditions)
        # Assumes format: (if cond0 conseq0 elif cond1 conseq1 else conseq2)
        args_str_lst = [arg.to_string() for arg in self.input_args]
        args_str = f"if {args_str_lst[0]} {args_str_lst[1]} "
        args_str += f"elif {args_str_lst[2]} {args_str_lst[3]} "
        args_str += f"else {args_str_lst[4]}"
        return f"({self.concept.to_string()} {args_str})" 

# TODO: figure out how this should fit in with Expression
class Concept:
    def __init__(self, input_types=None, output_type=None, func=None, val_input_args_func=None):
        self.input_types = input_types
        self.output_type = output_type
        self.func = func

        # This is a function that can validate the input args given
        # For example, for nth, we want to make sure that the index is <= the length of the list
        # If val_input_args is not supplied, set it to always return True
        if val_input_args_func is None:
            val_input_args_func = lambda *args: True
        self.val_input_args_func = val_input_args_func

    def is_primitive(self): 
        return self.primitive

    def to_string(self):
        pass 
    
    def __str__(self):
        return self.to_string()
    
    def val_input_args(self, input_args):
        return self.val_input_args_func(*input_args)

class Primitive(Concept):
    def __init__(self, val, output_type=None):
        # TODO: hacky, but allow overriding type for if inp variable like "x_" is supposed to have a particular type, we want to filter args by that type in enumerating possible programs 
        if output_type is None:
            output_type=type(val)
        super().__init__(output_type=output_type)
        self.value = val
        
        self.primitive = True

    def to_string(self):
        # TODO: hacky, but cannot parse commas yet
        return str(self.value).replace(',', '')

class Function(Concept):
    def __init__(self, name, is_symmetric=False, **kwargs):
        super().__init__(**kwargs)

        self.primitive = False
        self.name = name
        # Default to false when takes just one input
        self.is_symmetric = is_symmetric

    def to_string(self):
        return self.name
