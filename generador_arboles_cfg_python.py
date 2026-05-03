"""
Práctica POO - Generador de Derivación, Árbol de Derivación y AST
Lenguaje: Python
Interfaz gráfica: Tkinter
Librerías: nltk, matplotlib, networkx

Instalación necesaria:
pip install nltk matplotlib networkx

Ejecución:
python generador_arboles_cfg.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import nltk
import networkx as nx
import matplotlib.pyplot as plt
from nltk import CFG, ChartParser, Tree


class GrammarModel:
    """Clase encargada de cargar y validar la gramática."""

    def __init__(self, grammar_text):
        self.grammar_text = grammar_text
        self.grammar = CFG.fromstring(grammar_text)
        self.parser = ChartParser(self.grammar)

    def parse_expression(self, expression):
        tokens = expression.strip().split()
        trees = list(self.parser.parse(tokens))
        if not trees:
            return None
        return trees[0]


class DerivationGenerator:
    """Clase encargada de generar derivaciones por izquierda o derecha."""

    def __init__(self, parse_tree):
        self.parse_tree = parse_tree

    def left_derivation(self):
        steps = []
        current = [self.parse_tree.label()]
        steps.append(" ".join(current))
        self._derive(self.parse_tree, current, steps, left=True)
        return steps

    def right_derivation(self):
        steps = []
        current = [self.parse_tree.label()]
        steps.append(" ".join(current))
        self._derive(self.parse_tree, current, steps, left=False)
        return steps

    def _derive(self, node, current, steps, left=True):
        if isinstance(node, str):
            return

        symbol = node.label()
        replacement = []

        for child in node:
            if isinstance(child, Tree):
                replacement.append(child.label())
            else:
                replacement.append(child)

        index = self._find_symbol(current, symbol, left)
        if index != -1:
            current.pop(index)
            for item in reversed(replacement):
                current.insert(index, item)
            steps.append(" ".join(current))

        children = list(node)
        if not left:
            children = reversed(children)

        for child in children:
            if isinstance(child, Tree):
                self._derive(child, current, steps, left)

    def _find_symbol(self, current, symbol, left=True):
        if left:
            for i, item in enumerate(current):
                if item == symbol:
                    return i
        else:
            for i in range(len(current) - 1, -1, -1):
                if current[i] == symbol:
                    return i
        return -1


class ASTGenerator:
    """Clase encargada de simplificar el árbol de derivación para crear un AST."""

    OPERATORS = {"+", "-", "*", "/"}
    OMIT = {"(", ")"}

    def generate_ast(self, tree):
        return self._simplify(tree)

    def _simplify(self, node):
        if isinstance(node, str):
            if node in self.OMIT:
                return None
            return Tree(node, [])

        simplified_children = []
        for child in node:
            simplified = self._simplify(child)
            if simplified is not None:
                simplified_children.append(simplified)

        if len(simplified_children) == 1:
            return simplified_children[0]

        if len(simplified_children) == 3:
            left, middle, right = simplified_children
            if middle.label() in self.OPERATORS:
                return Tree(middle.label(), [left, right])

        return Tree(node.label(), simplified_children)


class TreeVisualizer:
    """Clase encargada de graficar árboles usando NetworkX y Matplotlib."""

    def show_tree(self, tree, title):
        graph = nx.DiGraph()
        labels = {}
        self._add_nodes(graph, labels, tree)

        pos = self._hierarchy_pos(graph, list(graph.nodes)[0])
        plt.figure(figsize=(10, 6))
        nx.draw(
            graph,
            pos,
            labels=labels,
            with_labels=True,
            node_size=1800,
            node_shape="o",
            font_size=10,
            arrows=False,
        )
        plt.title(title)
        plt.show()

    def _add_nodes(self, graph, labels, tree, parent=None, counter=[0]):
        node_id = counter[0]
        counter[0] += 1

        if isinstance(tree, Tree):
            label = tree.label()
        else:
            label = str(tree)

        graph.add_node(node_id)
        labels[node_id] = label

        if parent is not None:
            graph.add_edge(parent, node_id)

        if isinstance(tree, Tree):
            for child in tree:
                self._add_nodes(graph, labels, child, node_id, counter)

        return node_id

    def _hierarchy_pos(self, graph, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
        children = list(graph.successors(root))
        if not children:
            return {root: (xcenter, vert_loc)}

        pos = {root: (xcenter, vert_loc)}
        dx = width / len(children)
        next_x = xcenter - width / 2 - dx / 2

        for child in children:
            next_x += dx
            pos.update(
                self._hierarchy_pos(
                    graph,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=next_x,
                )
            )
        return pos


class CFGApp:
    """Clase principal de la interfaz gráfica."""

    def __init__(self, root):
        self.root = root
        self.root.title("Generador de Derivación, Árbol de Derivación y AST")
        self.root.geometry("1050x700")

        self.parse_tree = None
        self.ast_tree = None
        self.visualizer = TreeVisualizer()

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            main_frame,
            text="Generador de Árbol de Derivación y AST para Gramáticas Libres de Contexto",
            font=("Arial", 14, "bold"),
        )
        title.pack(pady=5)

        input_frame = ttk.LabelFrame(main_frame, text="Entrada", padding=10)
        input_frame.pack(fill=tk.X, pady=5)

        ttk.Label(input_frame, text="Gramática CFG:").pack(anchor="w")
        self.grammar_text = tk.Text(input_frame, height=7)
        self.grammar_text.pack(fill=tk.X)

        self.grammar_text.insert(
            tk.END,
            """S -> E
E -> E '+' T | T
T -> T '*' F | F
F -> '(' E ')' | 'id'""",
        )

        ttk.Label(input_frame, text="Expresión objetivo, separada por espacios:").pack(anchor="w", pady=(10, 0))
        self.expression_entry = ttk.Entry(input_frame)
        self.expression_entry.pack(fill=tk.X)
        self.expression_entry.insert(0, "id + id * id")

        options_frame = ttk.Frame(input_frame)
        options_frame.pack(fill=tk.X, pady=10)

        self.derivation_type = tk.StringVar(value="left")
        ttk.Radiobutton(
            options_frame,
            text="Derivación por la izquierda",
            variable=self.derivation_type,
            value="left",
        ).pack(side=tk.LEFT, padx=5)

        ttk.Radiobutton(
            options_frame,
            text="Derivación por la derecha",
            variable=self.derivation_type,
            value="right",
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(options_frame, text="Generar", command=self.generate).pack(side=tk.LEFT, padx=10)
        ttk.Button(options_frame, text="Ver árbol de derivación", command=self.show_derivation_tree).pack(side=tk.LEFT, padx=10)
        ttk.Button(options_frame, text="Ver AST", command=self.show_ast_tree).pack(side=tk.LEFT, padx=10)

        output_frame = ttk.LabelFrame(main_frame, text="Salida", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.output_text = tk.Text(output_frame, height=20)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def generate(self):
        try:
            grammar_input = self.grammar_text.get("1.0", tk.END).strip()
            expression = self.expression_entry.get().strip()

            grammar_model = GrammarModel(grammar_input)
            self.parse_tree = grammar_model.parse_expression(expression)

            if self.parse_tree is None:
                messagebox.showerror("Error", "La expresión no pertenece a la gramática ingresada.")
                return

            derivation_generator = DerivationGenerator(self.parse_tree)

            if self.derivation_type.get() == "left":
                steps = derivation_generator.left_derivation()
                derivation_title = "Derivación por la izquierda"
            else:
                steps = derivation_generator.right_derivation()
                derivation_title = "Derivación por la derecha"

            ast_generator = ASTGenerator()
            self.ast_tree = ast_generator.generate_ast(self.parse_tree)

            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, derivation_title + "\n")
            self.output_text.insert(tk.END, "=" * 50 + "\n\n")

            for i, step in enumerate(steps, start=1):
                self.output_text.insert(tk.END, f"Paso {i}: {step}\n")

            self.output_text.insert(tk.END, "\nÁrbol de derivación en formato texto:\n")
            self.output_text.insert(tk.END, str(self.parse_tree) + "\n")

            self.output_text.insert(tk.END, "\nÁrbol de Sintaxis Abstracta AST en formato texto:\n")
            self.output_text.insert(tk.END, str(self.ast_tree) + "\n")

            messagebox.showinfo("Resultado", "Derivación, árbol y AST generados correctamente.")

        except Exception as error:
            messagebox.showerror("Error", str(error))

    def show_derivation_tree(self):
        if self.parse_tree is None:
            messagebox.showwarning("Advertencia", "Primero debes generar el resultado.")
            return
        self.visualizer.show_tree(self.parse_tree, "Árbol de Derivación")

    def show_ast_tree(self):
        if self.ast_tree is None:
            messagebox.showwarning("Advertencia", "Primero debes generar el resultado.")
            return
        self.visualizer.show_tree(self.ast_tree, "Árbol de Sintaxis Abstracta AST")


if __name__ == "__main__":
    root = tk.Tk()
    app = CFGApp(root)
    root.mainloop()
