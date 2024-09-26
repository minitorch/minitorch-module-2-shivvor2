from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Set, Dict

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    vals_xi_epsilon = [val + epsilon if i == arg else val for i, val in enumerate(vals)]
    f_x_plus_h: float = f(*vals_xi_epsilon)
    f_x: float = f(*vals)
    derivative = (f_x_plus_h - f_x)/ epsilon
    
    return derivative


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # Assume no loops
    ordered_var_list: List[Variable] = []
    visited_vars: Set[Variable] = set()
    
    
    def depth_first_search(var: Variable) -> None:
        if (var.unique_id in visited_vars) or var.is_constant():
            return
        for parent in var.parents:
            depth_first_search(parent)
        visited_vars.add(var.unique_id)
        ordered_var_list.insert(0, var)
    
    depth_first_search(variable)
    
    return ordered_var_list
    
            


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    sorted_vars: Iterable[Variable] = topological_sort(variable)
    accumulated_d_vals: Dict[int, Any] = {}
    for var in sorted_vars:
        accumulated_d_vals[var.unique_id] = 0.0
    
    accumulated_d_vals[variable.unique_id] = deriv
    
    for var in sorted_vars:
        var_deriv = accumulated_d_vals[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(var_deriv)
        else:
            var_chain = var.chain_rule(var_deriv)
            for input, input_deriv in var_chain:
                accumulated_d_vals[input.unique_id] += input_deriv
        
        

    


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
