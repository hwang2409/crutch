from __future__ import annotations
import networkx as nx
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
from rational import Rational

# Lightweight eDAG leveraging networkx.DiGraph
@dataclass
class Node:
	type: str  # 'VAR','CONST','OP'
	symbol: str
	value: Any = None
	op: Optional[str] = None
	precedence: int = 0
	is_unary: bool = False
	children: List[str] = field(default_factory=list)  # ordered child node ids

class EDAG:
	def __init__(self) -> None:
		self.g = nx.DiGraph()
		self.root: Optional[str] = None
		self._id = 0
	def _nid(self) -> str:
		self._id += 1
		return f"n{self._id}"
	def parse(self, expr: str) -> None:
		# Deprecated in favor of parser.parse_expression_edag
		self.g.clear(); self.root = None
		n = self._nid()
		self.g.add_node(n, data=Node('EXPR', expr))
		self.root = n
	def _to_float(self, value: Rational | int | float) -> float:
		if isinstance(value, Rational):
			return value.numerator() / value.denominator()
		if isinstance(value, int):
			return float(value)
		return value
	def _norm(self, v: Any) -> Any:
		if isinstance(v, int):
			return Rational(v,1)
		return v
	def eval(self, env: Dict[str, Any] | None = None) -> Any:
		if self.root is None:
			raise RuntimeError('no expression parsed')
		env = env or {}
		# Local import to avoid circular dependency at module load time
		from parser import parse_expression_edag as _parse_edag_expr
		fns: Dict[str, Any] = {
			'sin': math.sin,
			'cos': math.cos,
			'tan': math.tan,
			'exp': math.exp,
			'log': math.log,
			'ln': math.log,
			'sqrt': math.sqrt,
		}
		def eval_node(nid: str) -> Any:
			data: Node = self.g.nodes[nid]['data']
			if data.type == 'CONST':
				return data.value
			if data.type == 'VAR':
				if data.symbol not in env:
					raise KeyError(f"Variable '{data.symbol}' not in env")
				return self._norm(env[data.symbol])
			if data.type == 'OP':
				# Evaluate children in order
				values = [eval_node(cid) for cid in data.children]
				# Unary minus
				if data.op == '-' and data.is_unary:
					v = values[0]
					return -v if not isinstance(v, Rational) else -v
				# Binary arithmetic
				if data.op in ('+','-','*','/','^'):
					a, b = values
					if isinstance(a, float) or isinstance(b, float):
						fa, fb = self._to_float(a), self._to_float(b)
						if data.op == '+': return fa + fb
						if data.op == '-': return fa - fb
						if data.op == '*': return fa * fb
						if data.op == '/': return fa / fb
						if data.op == '^': return fa ** fb
					# Rational/int arithmetic
					if data.op == '+': return a + b
					if data.op == '-': return a - b
					if data.op == '*': return a * b
					if data.op == '/': return a / b
					if data.op == '^':
						if isinstance(b, int):
							return a ** b
						if isinstance(b, Rational) and b.is_int():
							return a ** b.to_int()
						return self._to_float(a) ** self._to_float(b)
				# Functions
				if data.op in fns:
					args = [self._to_float(v) for v in values]
					return fns[data.op](*args)
				raise ValueError(f"Unknown op {data.op}")
			if data.type == 'EXPR':
				# Parse and evaluate embedded expression text
				sub = _parse_edag_expr(data.symbol)
				return sub.eval(env)
			raise ValueError('Unknown node type')
		return eval_node(self.root)

	# -----------------
	# Stringification
	# -----------------
	def _node_to_string(self, nid: str) -> str:
		data: Node = self.g.nodes[nid]['data']
		if data.type == 'CONST':
			v = data.value
			if isinstance(v, Rational):
				return v.to_string()
			return ("%g" % v)
		if data.type == 'VAR':
			return data.symbol
		if data.type == 'OP':
			# prefer stored children; fallback to graph predecessors if missing
			child_ids = data.children if len(data.children) > 0 else list(self.g.predecessors(nid))
			# Precedences for printing
			prec = { 'NEG':5,'^':4,'*':3,'/':3,'+':2,'-':2 }
			# if op is missing, just render children directly
			if data.op is None:
				if len(child_ids) == 0:
					return ""
				if len(child_ids) == 1:
					return self._node_to_string(child_ids[0])
				return "*".join(self._node_to_string(cid) for cid in child_ids)
			def is_const_id(cid: str) -> tuple[bool, Optional[Rational]]:
				cd = self.g.nodes[cid]['data']
				if cd.type == 'CONST' and isinstance(cd.value, Rational):
					return True, cd.value
				return False, None
			def wrap(child: str, parent_op: str, is_right: bool=False) -> str:
				cd = self.g.nodes[child]['data']
				if cd.type != 'OP':
					return self._node_to_string(child)
				cop = cd.op if not cd.is_unary else 'NEG'
				need = (prec.get(cop,6) < prec.get(parent_op,6)) or (is_right and parent_op == '^' and prec.get(cop,6) == prec.get(parent_op,6))
				s = self._node_to_string(child)
				return f"({s})" if need else s
			# unary minus
			if data.is_unary and data.op == '-' and len(child_ids) >= 1:
				arg = self._node_to_string(child_ids[0])
				return f"-{arg}"
			# binary infix (support n-ary fallback to avoid bare operators)
			if data.op in ('+','-','*','/','^'):
				k = len(child_ids)
				if k >= 2:
					# Minimal-safe printing (no aggressive folding)
					if data.op == '*':
						parts = [wrap(cid, '*') for cid in child_ids]
						# if any factor is zero, whole product is zero
						if any(p == "0" for p in parts):
							return "0"
						# safe cancellation: remove matching X and 1/X pairs
						to_remove = set()
						for i, pi in enumerate(parts):
							if pi.startswith("1/"):
								x = pi[2:]
								for j, pj in enumerate(parts):
									if j != i and pj == x:
										to_remove.add(i); to_remove.add(j)
										break
						if to_remove:
							parts = [p for idx,p in enumerate(parts) if idx not in to_remove]
						parts = [p for p in parts if p != "1" and p != ""]
						if len(parts) == 0:
							return "1"
						if len(parts) == 1:
							return parts[0]
						return "*".join(parts)
					elif data.op == '+':
						parts = [wrap(cid, '+') for cid in child_ids]
						parts = [p for p in parts if p != "0" and p != ""]
						if len(parts) == 0:
							return "0"
						if len(parts) == 1:
							return parts[0]
						return "+".join(parts)
					elif data.op == '-':
						is_c1, r1 = is_const_id(child_ids[1])
						if is_c1 and r1 == Rational(0,1):
							return wrap(child_ids[0], '-')
						is_c0, r0 = is_const_id(child_ids[0])
						if is_c0 and is_c1:
							return (r0 - r1).to_string()
						return f"{wrap(child_ids[0], '-')} - {wrap(child_ids[1], '-', is_right=True)}"
				if data.op == '/':
					left = wrap(child_ids[0], '/')
					right = wrap(child_ids[1], '/', is_right=True)
					if left == right:
						return "1"
					if left == "0":
						return "0"
					if right == "1":
						return left
					return f"{left}/{right}"
				if data.op == '^':
					left = wrap(child_ids[0], '^')
					right = wrap(child_ids[1], '^', is_right=True)
					if right == "1":
						return left
					if right == "0":
						return "1"
					return f"{left}^{right}"
				if k == 1:
					child = wrap(child_ids[0], data.op, is_right=False)
					return f"{data.op}{child}"
				return data.op
			# functions (n-ary)
			args = ",".join(self._node_to_string(cid) for cid in child_ids)
			return f"{data.op}({args})"
		return data.symbol
	def to_string(self) -> str:
		if self.root is None:
			return ''
		return self._node_to_string(self.root)
	def to_latex(self) -> str:
		return self.to_string()

	# -----------------
	# Symbolic differentiation
	# -----------------
	def _add_const(self, v: Any) -> str:
		n = self._nid()
		self.g.add_node(n, data=Node('CONST', str(v), value=(v if isinstance(v, Rational) else v)))
		return n
	def _add_var(self, name: str) -> str:
		n = self._nid()
		self.g.add_node(n, data=Node('VAR', name))
		return n
	def _add_expr_str(self, s: str) -> str:
		n = self._nid()
		self.g.add_node(n, data=Node('EXPR', s))
		return n
	def _add_op(self, op: str, children: List[str], is_unary: bool=False) -> str:
		n = self._nid()
		node = Node('OP', op, op=op, is_unary=is_unary, children=children)
		self.g.add_node(n, data=node)
		for c in children:
			self.g.add_edge(c, n)
		return n
	def _copy_subdag(self, nid: str, out: 'EDAG', memo: Dict[str,str]) -> str:
		if nid in memo: return memo[nid]
		data: Node = self.g.nodes[nid]['data']
		if data.type == 'CONST':
			n = out._nid(); out.g.add_node(n, data=Node('CONST', data.symbol, value=data.value)); memo[nid]=n; return n
		if data.type == 'VAR':
			n = out._nid(); out.g.add_node(n, data=Node('VAR', data.symbol)); memo[nid]=n; return n
		# OP
		child_copies = [ self._copy_subdag(c, out, memo) for c in data.children ]
		n = out._nid(); out.g.add_node(n, data=Node('OP', data.op, op=data.op, is_unary=data.is_unary, children=child_copies))
		for c in child_copies: out.g.add_edge(c, n)
		memo[nid]=n
		return n
	def derivative(self, var: str) -> 'EDAG':
		if self.root is None:
			raise RuntimeError('no expression parsed')
		out = EDAG()
		one = Rational(1,1); zero = Rational(0,1); two = Rational(2,1)
		def d(nid: str) -> str:
			data: Node = self.g.nodes[nid]['data']
			if data.type == 'CONST':
				return out._add_const(zero)
			if data.type == 'VAR':
				return out._add_const(one if data.symbol == var else zero)
			if data.type == 'OP':
				# unary minus
				if data.is_unary and data.op == '-':
					return out._add_op('-', [ d(data.children[0]) ], is_unary=True)
				# functions (chain rule)
				if data.op in ('sin','cos','tan','exp','log','ln','sqrt','asin','acos','atan'):
					u = data.children[0]
					du = d(u)
					if data.op == 'sin':
						fprime = out._add_op('cos', [ out._add_expr_str(self._node_to_string(u)) ])
					elif data.op == 'cos':
						fprime = out._add_op('-', [ out._add_op('sin', [ out._add_expr_str(self._node_to_string(u)) ]) ], is_unary=True)
					elif data.op == 'tan':
						# derivative is 1/cos(u)^2
						c = out._add_op('cos', [ out._add_expr_str(self._node_to_string(u)) ])
						fprime = out._add_op('^', [ c, out._add_const(Rational(2,1)) ])
						fprime = out._add_op('/', [ out._add_const(one), fprime ])
					elif data.op == 'exp':
						fprime = out._add_op('exp', [ out._add_expr_str(self._node_to_string(u)) ])
					elif data.op in ('log','ln'):
						fprime = out._add_op('/', [ out._add_const(one), out._add_expr_str(self._node_to_string(u)) ])
					elif data.op == 'sqrt':
						fprime = out._add_op('/', [ out._add_const(one), out._add_op('*', [ out._add_const(Rational(2,1)), out._add_op('sqrt', [ out._add_expr_str(self._node_to_string(u)) ]) ]) ])
					elif data.op == 'asin':
						# 1/sqrt(1-u^2)
						one_minus = out._add_op('-', [ out._add_const(one), out._add_op('^', [ out._add_expr_str(self._node_to_string(u)), out._add_const(Rational(2,1)) ]) ])
						fprime = out._add_op('/', [ out._add_const(one), out._add_op('sqrt', [ one_minus ]) ])
					elif data.op == 'acos':
						# -1/sqrt(1-u^2)
						one_minus = out._add_op('-', [ out._add_const(one), out._add_op('^', [ out._add_expr_str(self._node_to_string(u)), out._add_const(Rational(2,1)) ]) ])
						fprime = out._add_op('-', [ out._add_op('/', [ out._add_const(one), out._add_op('sqrt', [ one_minus ]) ]) ], is_unary=True)
					elif data.op == 'atan':
						# 1/(1+u^2)
						den = out._add_op('+', [ out._add_const(one), out._add_op('^', [ out._add_expr_str(self._node_to_string(u)), out._add_const(Rational(2,1)) ]) ])
						fprime = out._add_op('/', [ out._add_const(one), den ])
					else:
						raise ValueError('Unsupported function')
					return out._add_op('*', [ fprime, du ])
				# binary ops
				if data.op in ('+','-'):
					return out._add_op(data.op, [ d(data.children[0]), d(data.children[1]) ])
				if data.op == '*':
					u,v = data.children
					# Use textual copies to avoid identity/copy issues
					u_txt = out._add_expr_str(self._node_to_string(u))
					v_txt = out._add_expr_str(self._node_to_string(v))
					return out._add_op('+', [ out._add_op('*', [ d(u), v_txt ]), out._add_op('*', [ u_txt, d(v) ]) ])
				if data.op == '/':
					u,v = data.children
					num = out._add_op('-', [ out._add_op('*', [ d(u), out._copy_subdag(v, out, {}) ]), out._add_op('*', [ out._copy_subdag(u, out, {}), d(v) ]) ])
					den = out._add_op('^', [ out._copy_subdag(v, out, {}), out._add_const(two) ])
					return out._add_op('/', [ num, den ])
				if data.op == '^':
					a,b = data.children
					# Compute derivatives of children to detect constancy w.r.t var
					da = d(a)
					db = d(b)
					def is_zero_n(nid: str) -> bool:
						nd = out.g.nodes[nid]['data']
						return nd.type == 'CONST' and isinstance(nd.value, Rational) and nd.value.is_zero()
					const_a = is_zero_n(da)
					const_b = is_zero_n(db)
					# Prefer exponent as the child with zero derivative (constant), base as the other
					if const_b and not const_a:
						u, v = a, b; du, dv = da, db
					elif const_a and not const_b:
						u, v = b, a; du, dv = db, da
					else:
						u, v = a, b; du, dv = da, db
					vd: Node = self.g.nodes[v]['data']
					# If exponent is constant -> n*u^(n-1)*du
					if vd.type == 'CONST':
						nval = vd.value
						if isinstance(nval, Rational) and nval.is_int():
							ncoef = out._add_const(Rational(nval.to_int(),1))
							u_pow = out._add_op('^', [ out._add_expr_str(self._node_to_string(u)), out._add_const(Rational(nval.to_int()-1,1)) ])
							return out._add_op('*', [ ncoef, out._add_op('*', [ u_pow, du ]) ])
					# general a^b: a^b * ( db*log(a) + b*(da/a) )
					u_copy = out._add_expr_str(self._node_to_string(u))
					v_copy = out._add_expr_str(self._node_to_string(v))
					apowb = out._add_op('^', [ u_copy, v_copy ])
					term1 = out._add_op('*', [ dv, out._add_op('log', [ out._add_expr_str(self._node_to_string(u)) ]) ])
					term2 = out._add_op('*', [ out._add_expr_str(self._node_to_string(v)), out._add_op('/', [ du, out._add_expr_str(self._node_to_string(u)) ]) ])
					return out._add_op('*', [ apowb, out._add_op('+', [ term1, term2 ]) ])
				raise ValueError('Unsupported op')
			raise ValueError('Unknown node type')
		out.root = d(self.root)
		return out
	def to_string(self) -> str:
		if self.root is None:
			return ''
		return self._node_to_string(self.root)
	def to_latex(self) -> str:
		return self.to_string()
