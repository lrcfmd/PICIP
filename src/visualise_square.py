import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
import matplotlib.pyplot as plt
import math


class Square:
    """
    Plotly visualiser for a two-dimensional constrained phase field.

    Use this class when ``phase_field.constrained_dim == 2`` (e.g. a ternary
    system with one charge constraint, or any system whose constrained space is
    a 2-D simplex / polygon).  For three-dimensional constrained spaces use
    :class:`Cube` instead.

    The constructor immediately builds the base figure — corner markers and
    edge lines — by calling :meth:`create_fig_corners`.  All subsequent
    ``plot_*`` methods add traces on top of this base.  Call :meth:`show` (or
    ``show(return_fig=True)``) when you are ready to render.

    Parameters
    ----------
    phase_field : Phase_Field
        Must have ``constrained_dim == 2``, plus ``corners``, ``edges``, and
        ``corner_compositions`` populated (as set by
        ``Phase_Field._find_vertices_and_edges``).
    notebook : bool, optional
        If ``True``, renders with the ``notebook_connected`` Plotly renderer
        (needed inside Jupyter). Default ``False``.

    Raises
    ------
    Exception
        If ``phase_field.constrained_dim != 2``.
    """

    def __init__(self, phase_field, notebook=False):
        """
        Build the base figure (corners + edges) for a 2-D constrained space.

        Parameters
        ----------
        phase_field : Phase_Field
            Must have ``constrained_dim == 2``, ``corners``, ``edges``, and
            ``corner_compositions``.
        notebook : bool, optional
            Pass ``True`` when running inside a Jupyter notebook so that the
            correct Plotly renderer is selected. Default ``False``.
        """
        d=phase_field.constrained_dim
        if d!=2:
            if d==3:
                e='Phase field dimension is 3, use the Cube plotting class'
            else:
                e=f'Phase field dimension is {d}, this is not plottable'
            raise Exception(e)
        self.phase_field = phase_field
        self.resolution = phase_field.resolution
        if hasattr(phase_field, 'precursor_corners'):
            self.create_fig_corners(
                phase_field.convert_to_standard_basis(phase_field.precursor_corners),
                phase_field.precursor_edges,
                phase_field.get_composition_strings(phase_field.precursor_corners)
            )
        else:
            self.create_fig_corners(
                phase_field.corners,
                phase_field.edges,
                phase_field.corner_compositions
            )
        self.notebook=notebook

    def set_corner_text_position(self, corners):
        """
        Determine a suitable text position for corner labels.

        Parameters
        ----------
        corners : array-like
            List of corner coordinates in the standard basis.
        """
        text_position = []
        for x in corners:
            x_c = self.phase_field.convert_to_constrained_basis(x)
            # Determine horizontal alignment
            if abs(x_c[0] - self.xmin) < abs(x_c[0] - self.xmax):
                x_x = 'left'
            else:
                x_x = 'right'

            # Determine vertical alignment
            if abs(x_c[1] - self.ymin) < abs(x_c[1] - self.ymax):
                x_y = 'bottom'
            else:
                x_y = 'top'

            # Adjustments for corners on boundaries
            if x_c[0] == self.xmin or x_c[0] == self.xmax:
                if x_c[1] != self.ymin and x_c[1] != self.ymax:
                    x_y = 'middle'
            if x_c[1] == self.ymin or x_c[1] == self.ymax:
                if x_c[0] != self.xmin and x_c[0] != self.xmax:
                    x_x = 'center'

            text_position.append(x_y + " " + x_x)
        self.corner_text_position = text_position

    def create_fig_corners(self, corners, edges, labels, s=15, c='black'):
        """
        Create a figure with corners and edges.

        Parameters
        ----------
        corners : array-like
            List of corner coordinates in the standard basis.
        edges : list of tuples
            Connectivity of corners (e.g., [[0,1],[1,2],...]) to form edges.
        labels : list of str
            Labels for each corner.
        s : int, optional
            Size of corner markers, by default 10.
        c : str, optional
            Color of the corner markers, by default 'black'.

        Notes
        -----
        This creates a scatter plot with corners and lines for edges.
        """
        corners_c = self.phase_field.convert_to_constrained_basis(corners)
        df = pd.DataFrame(data=corners_c, columns=['x', 'y'])
        self.xmin = df['x'].min()
        self.xmax = df['x'].max()
        self.ymin = df['y'].min()
        self.ymax = df['y'].max()

        self.set_corner_text_position(corners)
        df['Label'] = labels

        hover_data = {col: False for col in df.columns}
        self.fig = px.scatter(
            df,
            x='x',
            y='y',
            text='Label',
            hover_data=hover_data
        )
        self.fig.update_traces(marker={'size': s, 'color': c},
                               textposition=self.corner_text_position)

        # Add edges as line segments
        for edge in edges:
            x = [corners_c[edge[0]][0], corners_c[edge[1]][0]]
            y = [corners_c[edge[0]][1], corners_c[edge[1]][1]]
            fig1 = go.Figure(
                data=go.Scatter(
                    x=x, y=y,
                    mode='lines',
                    line_color='black',
                    hoverinfo='skip',
                    showlegend=False
                )
            )
            for t in fig1.data:
                self.fig.add_trace(t)

    def show(self, title=None, show=True, save=None, legend_title=None, s=13,
             return_fig=False):
        """
        Finalise layout and render or export the figure.

        Call this after all ``plot_*`` methods have been called.  Layout
        styling (hidden axes, hover style, legend style) is applied here so
        that it always takes effect last.

        Parameters
        ----------
        title : str, optional
            Figure title text.  If ``None`` no title is set (a title may have
            already been set by :meth:`plot_prediction_results`).
        show : bool, optional
            If ``True`` (default), open the figure in the browser / notebook.
            Set to ``False`` when you only want to save or return the figure.
        save : str or None, optional
            Base filename (without extension) for HTML export.  Pass e.g.
            ``save='my_plot'`` to write ``my_plot.html``.  The file can be
            opened in any browser without a running Python server.
        legend_title : str or None, optional
            Title string rendered above the legend box.
        s : int, optional
            Global font size (Arial) for all text in the figure. Default 13.
        return_fig : bool, optional
            If ``True``, return the :class:`plotly.graph_objects.Figure` object
            instead of calling ``fig.show()``.  Useful for embedding the figure
            in a dashboard or post-processing it programmatically.

        Returns
        -------
        plotly.graph_objects.Figure or None
            The figure object when ``return_fig=True``; ``None`` otherwise.
        """
        self.fig.update_coloraxes(showscale=False)
        self.fig.update_xaxes(showticklabels=False, autorange=True, showgrid=False, zeroline=False, title='')
        self.fig.update_yaxes(showticklabels=False, autorange=True, showgrid=False, zeroline=False,
                              scaleanchor="x", scaleratio=1, title='')
        self.fig.update_layout(
            font=dict(family='Arial', size=s),
            hoverlabel=dict(
                bgcolor='rgba(25,25,25,0.88)',
                font_size=13,
                font_color='white',
                bordercolor='rgba(0,0,0,0)',
            ),
            legend=dict(
                bgcolor='rgba(255,255,255,0.92)',
                bordercolor='#CCCCCC',
                borderwidth=1,
                title=legend_title,
                font=dict(size=s + 2),
            ),
            plot_bgcolor='rgb(255,255,255)',
            paper_bgcolor='rgb(255,255,255)',
        )

        if return_fig:
            return self.fig
        if title is not None:
            self.fig.update_layout(title=dict(text=title))
        if show:
            if self.notebook:
                self.fig.show(renderer='notebook_connected')
            else:
                self.fig.show()
        if save is not None:
            self.fig.write_html(save + ".html")

    def plot_prediction_results(
            self, prediction, grid_s=3, point_s=14, show_comps_hover=True, comp_precision=2,
            colourscale='plasma', plot_samples=True, plot_knowns=True, plot_average_known=False,
            visible=True, plot_individual=True):
        """
        Plot the main PICIP result: probability density over the phase field
        with optional sample, known, and suggestion overlays.

        This is the primary method for visualising a :class:`Prediction` object.
        It adds traces in this order: probability density grid, per-sample
        individual densities (legend-only), sample points, known compositions,
        average-known marker, suggestion points.

        Parameters
        ----------
        prediction : Prediction
            Result returned by ``PICIP.run()``.
        grid_s : int, optional
            Marker size for the probability density grid points. Default 3.
        point_s : int, optional
            Marker size for sample / known / suggestion scatter points. Default 14.
        show_comps_hover : bool, optional
            If ``True`` (default), show the composition string in the hover
            tooltip for each grid point.
        comp_precision : int, optional
            Number of decimal places used in composition strings. Default 2.
        colourscale : str, optional
            Plotly colourscale name for the probability density. Default
            ``'plasma'``.
        plot_samples : bool, optional
            Overlay the measured sample composition(s) as white circle markers.
            Default ``True``.
        plot_knowns : bool, optional
            Overlay the known co-existing phase compositions as amber diamond
            markers (deduplicated across samples). Default ``True``.
        plot_average_known : bool, optional
            Overlay the weighted-average known composition (the centroid used
            internally by the algorithm) as an amber cross marker. Useful for
            debugging. Default ``False``.
        visible : bool or str, optional
            Plotly ``visible`` property for the probability density trace.
            ``True`` (default) shows it immediately; ``'legendonly'`` hides it
            on load but keeps the legend toggle.  Useful when overlaying
            multiple predictions on one figure.
        plot_individual : bool, optional
            When ``prediction`` combines multiple samples, also add each
            sample's individual density as a ``'legendonly'`` trace so the user
            can toggle them in the legend. Default ``True``.

        Notes
        -----
        The figure title is generated automatically from the prediction's
        sample names, compositions, and ``predicted_error`` values.
        """
        # Default title
        samples = prediction.samples
        if len(samples) == 1:
            s = samples[0]
            knowns_str = ' + '.join(
                f'{k.reduced_formula} ({w:.0%})'
                for k, w in zip(s.knowns, s.weights)
            )
            title = f'{s.name} | {s.composition.reduced_formula} | {knowns_str} | σ={s.predicted_error}'
        else:
            parts = [f'{s.name}: {s.composition.reduced_formula} (σ={s.predicted_error})' for s in samples]
            title = 'combined | ' + ' | '.join(parts)
        self.fig.update_layout(title=dict(text=title))

        self.plot_p(
                prediction.prob_density, s=grid_s, name=f'probability ({prediction.name})',
                show_comps_hover=show_comps_hover, comp_precision=comp_precision,
                colourscale=colourscale, visible=visible)

        if plot_individual and prediction.individual_densities is not None:
            for samp, p_ind in zip(prediction.samples, prediction.individual_densities):
                self.plot_p(p_ind, s=grid_s, name=f'probability ({samp.name})',
                            show_comps_hover=show_comps_hover, comp_precision=comp_precision,
                            colourscale=colourscale, visible='legendonly')

        if plot_samples:
            multi = len(prediction.samples) > 1
            samples_c, custom_labels = [], []
            for samp in prediction.samples:
                samples_c.append(samp.comp_constrained)
                if plot_knowns:
                    known_strs = [k.reduced_formula for k in samp.knowns]
                    custom_labels.append(f"<b>{samp.name}</b><br>knowns: {', '.join(known_strs)}")
                else:
                    custom_labels.append(f"<b>{samp.name}</b>")
            legend_name = 'samples' if multi else prediction.samples[0].name
            self.plot_points(samples_c, color='white', name=legend_name,
                             custom_labels=custom_labels, s=point_s,
                             border_color='#222222', border_width=2)

            if plot_knowns:
                # Collect unique knowns across all samples (deduplicate by reduced formula)
                seen = {}
                for samp in prediction.samples:
                    for k_comp, k_c in zip(samp.knowns, samp.knowns_constrained):
                        key = k_comp.reduced_formula
                        if key not in seen:
                            seen[key] = k_c

                if seen:
                    known_coords = np.array(list(seen.values()))
                    known_labels = [f"<b>{k}</b>" for k in seen.keys()]
                    self.plot_points(known_coords, color='#E8A838', name='knowns',
                                     custom_labels=known_labels, s=point_s,
                                     marker='diamond', border_color='#222222', border_width=2)

            if plot_average_known:
                avg_k_coords = np.array([s.average_k for s in prediction.samples])
                avg_k_labels = [f"<b>{s.name}</b><br>avg. known" for s in prediction.samples]
                self.plot_points(avg_k_coords, color='#E8A838', name='avg. known',
                                 custom_labels=avg_k_labels, s=point_s,
                                 marker='cross', border_color='#222222', border_width=2)

            if prediction.suggestions is not None:
                self.plot_suggestions(prediction.suggestions)

    def plot_detail_view(self, prediction,
                         show_projected_scatter=True,
                         omega_s=2, simplex_s=10, scatter_s=2, colourscale='plasma'):
        """
        Detailed diagnostic view of the algorithm internals for a single sample.

        Requires a single-sample :class:`Prediction` (i.e. produced by
        ``PICIP.run(sample_index=i)``).  Layers are added from bottom to top:

        1. **Projected distribution** — raw projected points coloured by their
           probability weight (semi-transparent, ``scatter_s`` sized).
        2. **Interpolated distribution** — the final ``griddata`` result
           evaluated on the omega grid (``omega_s`` sized).
        3. **Support assumption** — Dirichlet (or Gaussian for 1-known) PDF
           evaluated on the known simplex (``simplex_s`` sized).  ``None`` for
           a single-known sample (Gaussian path skips this layer).
        4. **Known compositions** — amber diamond markers.
        5. **Average known** — amber cross marker at the weighted centroid.
        6. **Sample composition** — white circle marker.
        7. **Suggestions** — if ``prediction.suggestions`` is set.

        Parameters
        ----------
        prediction : Prediction
            Must contain exactly one sample.  Raises ``ValueError`` if more
            than one sample is present.
        show_projected_scatter : bool, optional
            Include layer 1 (projected scatter).  Set ``False`` to keep the
            figure less cluttered. Default ``True``.
        omega_s : int, optional
            Marker size for the omega interpolated-distribution layer. Default 2.
        simplex_s : int, optional
            Marker size for the support-assumption (simplex) layer. Default 10.
        scatter_s : int, optional
            Marker size for the projected-distribution scatter. Default 2.
        colourscale : str, optional
            Plotly colourscale applied to all probability heatmap layers.
            Default ``'plasma'``.

        Raises
        ------
        ValueError
            If ``prediction`` contains more than one sample.
        Exception
            If the sample has no ``_simplex_points`` attribute (i.e. PICIP
            has not been run yet).
        """
        n_samples = len(prediction.samples)
        if n_samples > 1:
            raise ValueError(
                f"plot_detail_view requires a single-sample prediction "
                f"({n_samples} samples given). Run PICIP.run(sample_index) "
                "to get a prediction for one sample."
            )
        sample_index = 0

        sample = prediction.samples[sample_index]
        if not hasattr(sample, '_simplex_points'):
            raise Exception(
                'No detail data on this sample. Run PICIP.run() to populate it.')

        knowns_str = ' + '.join(
            f'{k.reduced_formula} ({w:.0%})'
            for k, w in zip(sample.knowns, sample.weights)
        )
        pf = self.phase_field
        subtitle = (f'ω: {len(pf.omega)} pts (res={pf.resolution}) | '
                    f'n_l={sample._n_rays}, n_p={sample._n_p}')
        title = (f'{sample.name} | {sample.composition.reduced_formula} | '
                 f'{knowns_str} | σ={sample.predicted_error}'
                 f'<br><sup>{subtitle}</sup>')
        self.fig.update_layout(title=dict(text=title))

        # 1 — Projected distribution: raw projected points (bottom layer)
        if show_projected_scatter:
            p = sample._projected_p
            self.plot_p(p, omega=sample._projected_points,
                        name='projected distribution', s=scatter_s,
                        colourscale=colourscale, opacity=0.5, cmin=0, cmax=p.max())

        # 2 — Interpolated distribution: griddata result on omega
        p = prediction.prob_density
        self.plot_p(p, name='interpolated distribution', s=omega_s,
                    colourscale=colourscale, opacity=0.6, cmin=0, cmax=p.max())

        # 3 — Support assumption: Dirichlet / Gaussian PDF on the known simplex
        if sample._simplex_points is not None:
            p = sample._simplex_pdf
            self.plot_p(p, omega=sample._simplex_points,
                        name='support assumption', s=simplex_s,
                        colourscale=colourscale, cmin=0, cmax=p.max())

        # 4 — Known compositions
        known_labels = [f'<b>{str(k.reduced_formula)}</b>' for k in sample.knowns]
        self.plot_points(sample.knowns_constrained, color='#E8A838', name='knowns',
                         custom_labels=known_labels, s=14, marker='diamond',
                         border_color='#222222', border_width=2)

        # 5 — Average known
        self.plot_points([sample.average_k], color='#E8A838', name='avg. known',
                         custom_labels=['<b>avg. known</b>'], s=14, marker='cross',
                         border_color='#222222', border_width=2)

        # 6 — Sample composition
        self.plot_points([sample.comp_constrained], color='white', name=sample.name,
                         custom_labels=[f'<b>{sample.name}</b>'], s=14,
                         border_color='#222222', border_width=2)

        if prediction.suggestions is not None:
            self.plot_suggestions(prediction.suggestions)

    def plot_plotting_df(self, name, color, s=15, df=None):
        """
        Plot the `phase_field.plotting_df` or a provided dataframe in 2D.

        Parameters
        ----------
        name : str
            Legend name for the plotted points.
        color : str
            Color of the points.
        s : int, optional
            Size of the points, by default None.
        df : pandas.DataFrame, optional
            The dataframe to plot. If None, uses `self.phase_field.plotting_df`.
            Must contain 'x' and 'y' columns.

        Raises
        ------
        Exception
            If `phase_field.plotting_df` does not exist and no df is provided.
            If 'x', 'y' columns are not present.
            If 'z' is in df (not allowed in 2D plotting).
        """
                                           
        if df is None:
            if not hasattr(self.phase_field, 'plotting_df'):
                raise Exception('Cannot plot if plotting_df does not exist.')
            df = self.phase_field.plotting_df

        if not {'x', 'y'}.issubset(df.columns):
            raise Exception('Dataframe must have x,y columns.')
        if 'z' in df.columns:
            raise Exception('Dataframe contains z but this is a 2D plotter.')

        hover_data = {
            'x': False,
            'y': False,
            'Composition': True,
        }

        fig1 = px.scatter(df, x='x', y='y', hover_data=hover_data)
        fig1.update_traces(marker_color=color, name=name, showlegend=True)

        if s is not None:
            fig1.update_traces(marker_size=s)

        for t in fig1.data:
            self.fig.add_trace(t)

    def plot_mesh(self, points, name=None):
        """
        Plot a polygonal mesh defined by a set of points in 2D.

        The first and last points are automatically connected to form a closed polygon.

        Parameters
        ----------
        points : array-like
            Coordinates of the mesh points in shape (n_points, 2).
        name : str, optional
            Name to be displayed in the legend, by default None.
        """
        x = list(points[:, 0])
        y = list(points[:, 1])
        # Close the polygon
        x.append(x[0])
        y.append(y[0])
        fig = go.Figure(go.Scatter(x=x, y=y))
        if name is not None:
            fig.update_traces(showlegend=True, name=name)
        for t in fig.data:
            self.fig.add_trace(t)

    def plot_points(self,
                    points,
                    color=None,
                    name=None,
                    legend_on_hover=True,
                    norm=None,
                    custom_labels=None,
                    s=15,
                    tol=1e-1,
                    marker=None,
                    border_color=None,
                    border_width=1.5):
        """
        Plot one or more composition points as a 2-D scatter trace.

        Accepts coordinates in either the standard (element-fraction) basis or
        the 2-D constrained basis.  The method validates charge neutrality,
        generates composition-string hover labels, and adds a single
        :class:`plotly.graph_objects.Scatter` trace.

        Call directly when you want to overlay arbitrary compositions that are
        not part of a :class:`Prediction` (e.g. candidate targets, literature
        values).

        Parameters
        ----------
        points : array-like, shape (N, 2) or (N, n_elements)
            Composition coordinates.  If the second dimension equals
            ``phase_field.nelements`` the points are automatically converted
            to the constrained basis.
        color : str or None, optional
            Marker fill colour (any Plotly colour string). ``None`` uses the
            Plotly default.
        name : str or None, optional
            Legend entry label.  ``None`` suppresses the legend entry.
        legend_on_hover : bool, optional
            Unused; kept for API backwards compatibility.
        norm : float or None, optional
            Passed to ``phase_field.get_composition_strings`` to normalise the
            displayed fractions (e.g. ``norm=1`` forces fractions to sum to 1).
            ``None`` uses the raw constrained-basis values.
        custom_labels : list of str or None, optional
            HTML strings prepended to each point's hover block (before the
            composition string).  Use ``<b>...</b>`` for bold text.  Must have
            the same length as ``points``.
        s : int, optional
            Marker size in pixels. Default 15.
        tol : float, optional
            Tolerance passed to ``phase_field.check_neutrality``. Default 1e-1.
        marker : str or None, optional
            Plotly marker symbol (e.g. ``'diamond'``, ``'cross'``).  ``None``
            uses the default circle.
        border_color : str or None, optional
            Marker border (line) colour.  ``None`` means no border.
        border_width : float, optional
            Marker border width. Default 1.5.

        Raises
        ------
        Exception
            If any point is not charge neutral within ``tol``, or if the point
            array has an unexpected shape.
        """
        points = np.array(points)
        if points.ndim == 1:
            points = np.array([points])
        if points.shape[1] == self.phase_field.nelements:
            points = self.phase_field.convert_to_constrained_basis(points)
        if points.shape[1] != self.phase_field.constrained_dim:
            raise Exception("Points must be provided in standard or constrained basis.")
        if points.shape[1] != 2:
            raise Exception("Constrained basis must be 2D.")

        points_s = self.phase_field.convert_to_standard_basis(points)
        self.phase_field.check_neutrality(points_s)

        formulas = (self.phase_field.get_composition_strings(points) if norm is None
                    else self.phase_field.get_composition_strings(points, norm=norm))
        # ensure formulas is always a list, not a bare string
        if isinstance(formulas, str):
            formulas = [formulas]

        # Build customdata and hovertemplate for clean, label-free hover text
        hover_cols = []
        hover_parts = []
        if custom_labels is not None:
            hover_cols.append(list(custom_labels))
            hover_parts.append('%{customdata[0]}')
        hover_cols.append(list(formulas))
        hover_parts.append(f'%{{customdata[{len(hover_cols) - 1}]}}')

        customdata = [list(row) for row in zip(*hover_cols)]
        hovertemplate = '<br>'.join(hover_parts) + '<extra></extra>'

        marker_dict = dict(size=s)
        if color is not None:
            marker_dict['color'] = color
        if marker is not None:
            marker_dict['symbol'] = marker
        if border_color is not None:
            marker_dict['line'] = dict(color=border_color, width=border_width)

        trace = go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode='markers',
            name=name,
            showlegend=name is not None,
            customdata=customdata,
            hovertemplate=hovertemplate,
            marker=marker_dict,
        )
        self.fig.add_trace(trace)

    def plot_line(self, start, end, c='black', w=2, name=None):
        """
        Plot a line between two points in 2D.

        Parameters
        ----------
        start : array-like
            Starting point (x, y).
        end : array-like
            Ending point (x, y).
        c : str, optional
            Line color, by default 'black'.
        w : int, optional
            Line width, by default 2.
        """
        fig1=go.Figure(
            go.Scatter(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                mode='lines',
                line={'color': c, 'width': w}
            )
        )
        if name is not None:
            fig1.update_traces(name=name, showlegend=True)
        else:
            fig1.update_traces(showlegend=False)
        for t in fig1.data:
            self.fig.add_trace(t)


    def draw_sample_line(self, sample_index=-1, w=10, c='black', legend=None):
        """
        Draw a line between two sample points defined by the phase field.

        The endpoints of the line are determined by the `phase_field.end_points` and 
        `phase_field.s` attributes.

        Parameters
        ----------
        sample_index : int, optional
            Index of the sample line to draw, by default -1 (the last one).
        w : int, optional
            The width of the line, by default 10.
        c : str, optional
            Color of the line, by default 'black'.
        legend : str, optional
            Name to show in the legend for this line, by default None.

        Raises
        ------
        AttributeError
            If `phase_field.end_points` or `phase_field.s` are not defined.
        """
        self.phase_field.set_end_points_from_s()
        ep = self.phase_field.end_points[sample_index]
        sp = self.phase_field.s[sample_index]

        columns = ['x', 'y']
        df = pd.DataFrame(columns=columns, data=[sp, ep])
        hover_dict = {'x': False, 'y': False}

        fig1 = px.line(df, x='x', y='y', hover_data=hover_dict)
        fig1.update_traces(line_width=w, line_color=c)

        if legend is not None:
            fig1.update_traces(name=legend, showlegend=True)

        for t in fig1.data:
            self.fig.add_trace(t)

    def plot_skline(self,color=None,n=None):
        """
        Plot sample points and known average points, and draw a line between them.

        This method uses:
        - `self.phase_field.s[-1]` as the sample points (plotted in red).
        - `self.phase_field.k[-1]` as the known average points (plotted in green).
        - Draws a sample line using `draw_sample_line` with a green color.

        Raises
        ------
        AttributeError
            If `phase_field.s` or `phase_field.k` are not defined.
        """
        if color is None:
            cs='red'
            ck='green'
            cl='green'
        else:
            cs=color
            ck='grey'
            cl=color
        if n is None:
            self.plot_points(self.phase_field.s[-1], color=cs, name='Sample')
            self.plot_points(self.phase_field.k[-1], color=ck, name='Average known')
            self.draw_sample_line(c=cl,legend='Estimated direction')
        else:
            self.plot_points(self.phase_field.s[-1], color=cs, name=f'Sample {n}')
            self.plot_points(self.phase_field.k[-1], color=ck)
            self.draw_sample_line(c=cl)


    def _filter_top_p(self, p, top, omega):
        """Return (omega_top, p_top) for the top `top` fraction of points by probability, sorted descending."""
        idx = np.argsort(p)[::-1]
        cutoff = max(1, int(np.ceil(top * len(p))))
        return omega[idx[:cutoff]], p[idx[:cutoff]]

    def plot_p_top(self, p, top=0.9, s=5, omega=None, name=None, opacity=1,
                   show_comps_hover=False, comp_precision=2, colourscale='plasma'):
        """
        Plot only the grid points that account for the top fraction of probability mass.

        Convenience wrapper around :meth:`plot_p` that filters ``omega`` and
        ``p`` to the highest-density subset before plotting.  Useful for
        de-cluttering sparse 2-D distributions.

        Parameters
        ----------
        p : array-like, shape (N,)
            Probability density values over the full grid.
        top : float, optional
            Fraction of total probability mass to retain, e.g. ``0.9`` keeps
            the 90 % highest-density points. Default ``0.9``.
        s : int, optional
            Marker size. Default 5.
        omega : array-like or None, optional
            Grid coordinates.  ``None`` uses ``self.phase_field.omega``.
        name : str or None, optional
            Legend label.
        opacity : float, optional
            Marker opacity. Default 1.
        show_comps_hover : bool, optional
            Show composition strings in hover tooltip. Default ``False``.
        comp_precision : int, optional
            Decimal places in composition strings. Default 2.
        colourscale : str, optional
            Plotly colourscale name. Default ``'plasma'``.
        """
        if omega is None:
            omega = self.phase_field.omega
        omega_top, p_top = self._filter_top_p(p, top, omega)
        self.plot_p(p_top, s=s, omega=omega_top, name=name, opacity=opacity,
                    show_comps_hover=show_comps_hover, comp_precision=comp_precision,
                    colourscale=colourscale)

    def plot_p(self, p, s=5, omega=None, name=None, opacity=1,
               show_comps_hover=False, comp_precision=2, colourscale='plasma',
               cmin=None, cmax=None, visible=True):
        """
        Plot a scalar probability field as a 2-D scatter coloured by ``p``.

        Typically called internally by :meth:`plot_prediction_results` and
        :meth:`plot_detail_view`, but can also be called directly to overlay
        an arbitrary probability array on the figure.

        Parameters
        ----------
        p : array-like, shape (N,)
            Scalar values that map to marker colour.  Usually the normalised
            probability density evaluated on ``omega``.
        s : int, optional
            Marker size. Default 5.
        omega : array-like, shape (N, 2), optional
            2-D constrained-basis coordinates of the points.  If ``None``,
            uses ``self.phase_field.omega`` (the standard grid).
        name : str or None, optional
            Legend entry label.  Pass ``None`` to suppress the legend entry.
        opacity : float, optional
            Marker opacity in [0, 1]. Default 1.
        show_comps_hover : bool, optional
            If ``True``, include the composition string in the hover tooltip.
            Slightly slower for large grids. Default ``False``.
        comp_precision : int, optional
            Decimal places in composition strings when ``show_comps_hover`` is
            ``True``. Default 2.
        colourscale : str, optional
            Plotly colourscale name. Default ``'plasma'``.
        cmin : float or None, optional
            Lower bound of the colour axis.  ``None`` uses the data minimum.
        cmax : float or None, optional
            Upper bound of the colour axis.  ``None`` uses the data maximum.
        visible : bool or str, optional
            Plotly ``visible`` property.  ``True`` (default) shows immediately;
            ``'legendonly'`` hides the trace but keeps the legend toggle.
        """
        if omega is None:
            omega = self.phase_field.omega

        if show_comps_hover:
            comp_strings = self.phase_field.get_composition_strings(
                omega, prec=comp_precision)
            hovertemplate = '<b>%{customdata[0]}</b><br>p=%{marker.color:.2e}<extra></extra>'
            customdata = [[c] for c in comp_strings]
        else:
            hovertemplate = 'p=%{marker.color:.2e}<extra></extra>'
            customdata = None

        trace = go.Scatter(
            x=omega[:, 0], y=omega[:, 1],
            mode='markers',
            marker=dict(
                color=p,
                colorscale=colourscale,
                cmin=cmin, cmax=cmax,
                size=s,
                opacity=opacity,
                showscale=False,
            ),
            name=name if name is not None else '',
            showlegend=name is not None,
            hovertemplate=hovertemplate,
            customdata=customdata,
            visible=visible,
        )
        self.fig.add_trace(trace)

    def plot_spread_result(self, result, s=14, plot_known_phases=True,
                           color='#4CAF50', name='spread compositions'):
        """
        Overlay spread composition points on the figure.

        Plots the suggested compositions from a :class:`SpreadResult` as
        circle markers.  Optionally also plots the fixed known phases that were
        used as constraints during spreading.

        Parameters
        ----------
        result : SpreadResult
            Object returned by ``Spread.run()``.
        s : int, optional
            Marker size. Default 14.
        plot_known_phases : bool, optional
            If ``True`` (default), also overlay the known phases used as fixed
            sites during spreading, styled as amber diamond markers.
        color : str, optional
            Fill colour for the spread points. Default green ``'#4CAF50'``.
        name : str, optional
            Legend label for the spread points. Default ``'spread compositions'``.
        """
        labels = [f'<b>{i + 1}</b>' for i in range(len(result.points))]
        self.plot_points(
            result.points, color=color, name=name,
            custom_labels=labels, s=s, marker='circle',
            border_color='#222222', border_width=2,
        )
        if plot_known_phases and hasattr(self.phase_field, 'knowns_constrained'):
            self.plot_points(
                self.phase_field.knowns_constrained,
                color='#E8A838', name='known phases',
                s=s, marker='diamond',
                border_color='#222222', border_width=2,
            )

    def plot_suggestions(self, suggestions, s=14):
        """
        Overlay next-experiment suggestion points on the figure.

        Styled according to the label stored in ``suggestions.labels``:

        * ``'mean'`` — white fill, amber (``#E8A838``) border; represents the
          posterior mean of the prediction.
        * ``'sampled'`` — amber fill, dark border; represents points drawn
          randomly from the posterior distribution.

        Called automatically by :meth:`plot_prediction_results` and
        :meth:`plot_detail_view` when ``prediction.suggestions`` is not
        ``None``.  Call directly if you want to overlay suggestions on a
        figure built manually with :meth:`plot_p` or :meth:`plot_points`.

        Parameters
        ----------
        suggestions : Suggestions
            Object returned by ``PICIP.suggest()``.  Must have
            ``constrained``, ``labels``, and ``elements`` attributes.
        s : int, optional
            Marker size. Default 14.
        """
        labels = np.array(suggestions.labels)

        mean_pts = suggestions.constrained[labels == 'mean']
        if len(mean_pts):
            self.plot_points(mean_pts, color='white', name='suggestion (mean)',
                             custom_labels=['<b>mean</b>'] * len(mean_pts),
                             s=s, marker='circle',
                             border_color='#E8A838', border_width=3)

        samp_pts = suggestions.constrained[labels == 'sampled']
        if len(samp_pts):
            samp_labels = [f'<b>sampled {i+1}</b>' for i in range(len(samp_pts))]
            self.plot_points(samp_pts, color='#E8A838', name='suggestions (sampled)',
                             custom_labels=samp_labels, s=s, marker='circle',
                             border_color='#222222', border_width=2)

