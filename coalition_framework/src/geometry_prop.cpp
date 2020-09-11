#include <math.h>
#include "geometry_prop.h"

//#include "Multiout.h"
#define VERBOSE 0
#define _USE_MATH_DEFINES 1
#define TEST_CODE 0

namespace geometry_prop
{

    void polylineToVecteurSegment2(const Polyline_2 poly, vector<Segment_2>& vec)
    {
        for (uint i = 0; i < poly.size(); i++)
        {
            Point_2 p1 = poly[i];
            if (i + 1 < poly.size())
            {
                Point_2 p2 = poly[i + 1];
                Segment_2 s;
                if (p1.x() <= p2.x())
                    s = Segment_2(p1, p2);
                else
                    s = Segment_2(p2, p1);
                vec.push_back(s);
            }
        }
    }


    bool intersectCercleCercle(const int sens,
                               const double X0,
                               const double Y0,
                               const double R0_carre,
                               const double X1,
                               const double Y1,
                               const double R1_carre,
                               double& X,
                               double& Y
                               )
    {
        bool trouve = false;

        // Cas où les cercles sont alignés sur la même hauteur Y
        if (Y0 == Y1)
        {
            // Intersections à coordonnées X identique
            X = (R1_carre - R0_carre - X1*X1 + X0*X0)
                / (2 * (X0 - X1));
            //y = [2.y1 + racine ( (-2.y1)² - 4.(x1²+ x² - 2.x1.x + y1² - R1²)  )] / 2
            double delta = 4 * Y1*Y1 - 4 * (X1*X1 + X*X - 2 * X1*X + Y1*Y1 - R1_carre);
            if (delta >= 0)
            {
                trouve = true;
                // force la solution la plus haute suivant Y (Y max)
                Y = (2 * Y1 + sqrt(delta)) / 2;
            }
            else
                trouve = false;
        }
        else
        {
            double N = (R1_carre - R0_carre - X1*X1 + X0*X0 - Y1*Y1 + Y0*Y0)
                / (2 * (Y0 - Y1));

            double A = ((X0 - X1) / (Y0 - Y1))*((X0 - X1) / (Y0 - Y1)) + 1;
            double B = 2 * Y0*((X0 - X1) / (Y0 - Y1)) - 2 * N*((X0 - X1) / (Y0 - Y1)) - 2 * X0;
            double C = X0*X0 + Y0*Y0 + N*N - R0_carre - 2 * Y0*N;

            double delta = B*B - 4 * A*C;

            if (delta > 0)
            {
                trouve = true;

                // Deux solutions, retient soit la plus à droite (sens==0)
                // soit à gauche (sens==1).
                // FIXME,BM: condition du pivot remplacé dans le code UTBM par:
                if ((X0 - X1) / (Y0 - Y1) <= 0)
                //if (sens == 0)
                    X = (-B + sqrt(delta)) / (2 * A);
                else
                    X = (-B - sqrt(delta)) / (2 * A);

                Y = N - X*((X0 - X1) / (Y0 - Y1));
            }
            else if (delta == 0)
            {
                // Solution double (tangent)
                trouve = true;
                X = -B / (2 * A);
                Y = N - X*((X0 - X1) / (Y0 - Y1));
            }
        }
        return trouve;
    }

    bool intersectCercleCercle(const Vector_2 vSensDirection,
                               const double X0,
                               const double Y0,
                               const double R0_carre,
                               const double X1,
                               const double Y1,
                               const double R1_carre,
                               double& X,
                               double& Y
                               )
    {
        bool trouve = false;

        // Cas où les cercles sont alignés sur la même hauteur Y
        if (Y0 == Y1)
        {
            // Intersections à coordonnées X identique
            X = (R1_carre - R0_carre - X1*X1 + X0*X0)
                / (2 * (X0 - X1));
            //y = [2.y1 + racine ( (-2.y1)² - 4.(x1²+ x² - 2.x1.x + y1² - R1²)  )] / 2
            double delta = 4 * Y1*Y1 - 4 * (X1*X1 + X*X - 2 * X1*X + Y1*Y1 - R1_carre);
            if (delta >= 0)
            {
                trouve = true;
                // force la solution la plus haute suivant Y (Y max)
                Y = (2 * Y1 + sqrt(delta)) / 2;
            }
            else
                trouve = false;
        }
        else
        {
            double N = (R1_carre - R0_carre - X1*X1 + X0*X0 - Y1*Y1 + Y0*Y0)
                / (2 * (Y0 - Y1));

            double A = ((X0 - X1) / (Y0 - Y1))*((X0 - X1) / (Y0 - Y1)) + 1;
            double B = 2 * Y0*((X0 - X1) / (Y0 - Y1)) - 2 * N*((X0 - X1) / (Y0 - Y1)) - 2 * X0;
            double C = X0*X0 + Y0*Y0 + N*N - R0_carre - 2 * Y0*N;

            double delta = B*B - 4 * A*C;

            if (delta > 0)
            {
                trouve = true;

                double XX1 = (-B - sqrt(delta)) / (2 * A);
                double YY1 = N - XX1*((X0 - X1) / (Y0 - Y1));

                double XX2 = (-B + sqrt(delta)) / (2 * A);
                double YY2 = N - XX2*((X0 - X1) / (Y0 - Y1));

                Point_2 P1(XX1, YY1);
                Point_2 P2(XX2, YY2);

                // Sens des points doit être conforme conforme
                if (vSensDirection * Vector_2(P1, P2) >= 0) {
                    X = P1.x();
                    Y = P1.y();
                }
                else {
                    X = P2.x();
                    Y = P2.y();
                }

            }
            else if (delta == 0)
            {
                // Solution double (tangent)
                trouve = true;
                X = -B / (2 * A);
                Y = N - X*((X0 - X1) / (Y0 - Y1));
            }
        }
        return trouve;
    }

    bool intersectCercleDroite(const Point_2 cRot,
                               const double d_carre,
                               const Segment_2 seg,
                               Segment_2& seg_cut)
    {
        // Par défaut : pas d'intersection
        bool trouve = false;
        // Droite support du segment:
        //    x = x0 + t*dx avec dx=x1-x0
        //    y = y0 + t*dy avec dy=y1-y0
        // Equation du cercle:
        //    (x-xc)^2 + (y-yc)^2 = R^2
        // substitution:
        //    (t*dx+Ox)^2 + (t*dy+Oy)^2 = R^2
        // avec Ox=x0-xc et Oy=y0-yc
        //    A*t^2 + B*t + C = 0
        //    A=dx^2+dy^2
        //    B=2*(dx.Ox+dy.Oy)
        //    C= Ox^2+Oy^2-R^2
        // => résolution pour t puis réinjection

        double dx = seg.target().x() - seg.source().x();
        double dy = seg.target().y() - seg.source().y();
        double Ox = seg.source().x() - cRot.x();
        double Oy = seg.source().y() - cRot.y();

        double A = dx*dx + dy*dy;
        double B = 2 * (dx*Ox + dy*Oy);
        double C = Ox*Ox + Oy*Oy - d_carre;

        double delta = B * B - 4 * A * C;

        if (delta > 0)
        {
            // Deux intersections avec la droite support
            trouve = true;

            // x = x0 + t*dx avec dx=x1-x0
            // y = y0 + t*dy avec dy=y1-y0

            // t1 à Xmin, T2 à Xmax
            double t1 = (-B - sqrt(delta)) / (2 * A);
            double t2 = (-B + sqrt(delta)) / (2 * A);

            // Définition du segment reliant les deux intersections
            Point_2 p1(seg.source().x() + t1*dx, seg.source().y() + t1*dy);
            Point_2 p2(seg.source().x() + t2*dx, seg.source().y() + t2*dy);
            seg_cut = Segment_2(p1, p2);

            // Réoriente le segment pour suivre la même direction que le segment
            // d'origine. On utilise un produit scalaire pour avoir le signe de
            // la direction
            if (seg_cut.to_vector()*seg.to_vector() < 0)
            {
                seg_cut = seg_cut.opposite();
            }
        }
        else if (delta == 0)
        {
            // cercle tangent au segment, un seul point double
            trouve = true;

            //    x = x0 + t*dx avec dx=x1-x0
            //    y = y0 + t*dy avec dy=y1-y0
            double t1 = -B / (2 * A);

            // Segment constitué d'un unique point double
            Point_2 p1(seg.source().x() + t1*dx, seg.source().y() + t1*dy);
            Point_2 p2(seg.source().x() + t1*dx, seg.source().y() + t1*dy);
            seg_cut = Segment_2(p1, p2);
        }
        return trouve;

    }//intersectCercleDroite

#if TEST_CODE
    bool intersectCercleDroite(const int pivot,
                               const double X0,
                               const double Y0,
                               const double d_carre,
                               const double a,
                               const double b,
                               double& XC,
                               double& YC
                               )
    {
        bool trouve = false;

        Segment_2 seg_cut;

        trouve = intersectCercleDroite(Point_2(X0, Y0),
                                       d_carre,
                                       Segment_2(Point_2(0, b), Point_2(1, a + b)),
                                       seg_cut);

        if (pivot == 0)
        {
            XC = seg_cut.target().x();
            YC = seg_cut.target().y();
        }
        else
        {
            XC = seg_cut.source().x();
            YC = seg_cut.source().y();
        }
        return trouve;

    }//intersectCercleDroite
#endif

    bool intersectCercleDroite(const int sens,
                               const double X0,
                               const double Y0,
                               const double d_carre,
                               const double a,
                               const double b,
                               double& XC,
                               double& YC
                               )
    {
        bool trouve = false;
        // Let's say you have the line y=a*x+b 
        //  and the circle (x−X0)^2 +(y−Y0)^2 = r^2
        // First, substitute y = a*x+b into circle to give
        // (x−X0)^2+(a*x+b−Y0)^2 = r^2.
        // Next, expand out both brackets, bring the r^2
        // over to the left, and collect like terms :
        // (a^2+1)*x^2 + 2a*(b−Y0)*x + (X0^2+(b-Y0)^2−r^2)=0.
        // Now solve for x as :
        // A*x^2+B*x+C=0
        double A = 1 + a * a;
        double B = 2 * a * (b - Y0) - 2 * X0;
        double C = X0 * X0 + (b - Y0) * (b - Y0) - d_carre;
        double delta = B * B - 4 * A * C;
        if (delta > 0)
        {
            trouve = true;
            // Deux solutions, retient soit la plus à l'arrière (sens==0)
            // soit à l'avant (sens==1).
            if (sens == 0)
                XC = (-B + sqrt(delta)) / (2 * A);
            else
                XC = (-B - sqrt(delta)) / (2 * A);
            // Réinjecte X pour Y
            YC = a * XC + b;
        }
        else if (delta == 0)
        {
            // Solution double (tangent)
            trouve = true;
            XC = -B / (2 * A);
            YC = a * XC + b;
        }
        return trouve;
    }

    bool intersectVerticaleCercle(const double X0,
                                  const double X1,
                                  const double Y1,
                                  const double R1_carre,
                                  double& X,
                                  double& Y
                                  )
    {
        bool trouve = false;
        // Distance en X, au carré
        double distanceX_square = (X0 - X1) * (X0 - X1);
        // Si contact, distance en X entre centre du cercle et droite doit être
        // inférieure au rayon.
        if (R1_carre >= distanceX_square)
        {
            trouve = true;
            // Si contact, ne peut être qu'à XC=X0
            X = X0;
            // Résoudre pour YC l'eq. du cercle:
            //   (XC-X1)^2+(YC-Y1)^2 = r^2.
            //    YC = (+/-)sqrt(r^2 - (XC-X1)^2) + Y1
            // Sélectionne toujours le Y max (signe +):
            Y = sqrt(R1_carre - distanceX_square) + Y1;
        }
        return trouve;
    }

    bool intersectVerticaleDroite(const double X0,
                                  const double Y0,
                                  const double a,
                                  const double b,
                                  double& XC,
                                  double& YC
                                  )
    {
        bool trouve = true;
        // D1: x=X0 => si contact, ne peut être qu'à XC=X0
        // D2: y=a*x+b => YC=a*XC+b
        XC = X0;
        YC = a * XC + b;

        return trouve;
    }//intersectVerticaleDroite

    Polygon_2 getClosedPolygon(const Polygon_2 & polygon)
    {
        Polygon_2 p(polygon.container().begin(), polygon.container().end());
        p.push_back(p.front());
        return p;
    }

    void extractVector(std::string& std_str, std::vector<Point_2> &pts)
    {
        // Parser dans un vecteur
        std::vector<double> v;
        boost::char_separator<char> sep(" ,\t\n");
        boost::tokenizer<boost::char_separator<char> > tok(std_str, sep);
        std::transform(tok.begin(), tok.end(), std::back_inserter(v), ToDouble());

        // Construire vecteur de Point_2
        for (std::vector<double>::iterator it = v.begin(); it != v.end(); it += 2)
        {
            pts.push_back(Point_2(*it, *(it + 1)));
        }
    }

    void extractContour(std::string& std_str, Polygon_2& p)
    {
        // Construire vecteur de Point_2
        std::vector<Point_2> pts;
        extractVector(std_str, pts);
        // Construire Polygon_2
        p = Polygon_2(pts.begin(), pts.end());
    }

    void extractChemin(std::string& std_str, Polyline_2& p)
    {
        // Construire vecteur de Point_2
        std::vector<Point_2> pts;
        extractVector(std_str, pts);
        // Construire Polygon_2
        p = Polyline_2(pts.begin(), pts.end());
    }

    bool verifyAndUpdatePolygon(Polygon_2& p)
    {

        // Verification des self-intersection (polygone mal formé)
        bool isSelfIntersect = p.selfintersects();
        if (isSelfIntersect)
        {
            cout << "Polygon mal formé !";
            return false;
        }
        // Polygone simple = non fermé (premier et dernier points sont différents)
        bool IsSimple = is_simple(p);
#if VERBOSE
        lout << "VOLUME p is";
        if (!IsSimple) lout << " not";
        lout << " simple." << std::endl;
#endif
        if (!IsSimple)
        {
            vector<Point_2>& v = p.container();
            v.pop_back();
        }

        bool IsClockwise = is_clockwise_oriented(p);
#if VERBOSE
        lout << "VOLUME is";
        if (!IsClockwise) lout << " not";
        lout << " clockwise oriented." << std::endl;
#endif
        if (!IsClockwise)
        {
#if VERBOSE
            lout << "VOLUME : reverse orientation !!" << std::endl;
#endif
            reverse_orientation(p);
        }
        return true;
    }

    bool verifyAndUpdatePolyline(Polyline_2& pol)
    {
        // Verification des self-intersection (polyline mal formé)
        bool isSelfIntersect = pol.selfintersects();
        if (isSelfIntersect)
        {
            cout << "Polyline mal formé !";
            return false;
        }

        //verifier l'orientation avant->arriere
        bool bad_orientation = false;
        for (uint i = 0; i < pol.size(); i++)
        {
            Point_2 p1 = pol[i];
            if (i + 1 < pol.size() && i + 2 < pol.size())
            {
                Point_2 p2 = pol[i + 1];
                Point_2 p3 = pol[i + 2];
                if (p2.x() < p1.x() && p3.x() < p2.x())
                {
                    bad_orientation = true;
                    break;
                }
            }
            else if (pol.size() == 2 && i + 1 < pol.size())
            {
                Point_2 p2 = pol[i + 1];
                if (p2.x() < p1.x())
                {
                    bad_orientation = true;
                    break;
                }
            }
        }
        if (bad_orientation)
        {
            //        lout << "reverse orientation !!" << std::endl;
            std::reverse(pol.begin(), pol.end());
        }
        return true;
    }

    void normaliseCroissantPolyline(Polyline_2& pol)
    {
        //verifier l'orientation avant->arriere
        bool bad_orientation = false;
        for (uint i = 0; i < pol.size(); ++i)
        {
            Point_2 p1 = pol[i];
            for (uint j = i + 1; j < pol.size(); ++j)
            {
                if (j < pol.size())
                {
                    Point_2 p2 = pol[j];
                    if (p2.x() > p1.x())
                    {
                        pol[i + 1] = p2;
                        break;
                    }
                    else
                    {
                        bad_orientation = true;
                        // Mieux que rien
                        pol[i + 1] = Point_2(p1.x() + 100, p2.y());
                    }
                }
            }
        }
    }

}//namespace geometry_prop

