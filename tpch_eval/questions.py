"""Natural-language versions of the 22 TPC-H queries.

Each question is the TPC-H business question with the standard validation
parameters filled in, plus the exact output columns (in order). Gold answers
come from running the official SQL in reference/qNN.sql on the same data.
"""

QUESTIONS: dict[int, str] = {
    1: """Pricing summary report. For all line items shipped on or before 1998-09-02, group by return flag and line status and report:
the return flag, the line status, total quantity, total extended price, total discounted price (extended price * (1 - discount)),
total charge (discounted price * (1 + tax)), average quantity, average extended price, average discount, and the number of line items.
Sort by return flag, then line status.
Output columns: l_returnflag, l_linestatus, sum_qty, sum_base_price, sum_disc_price, sum_charge, avg_qty, avg_price, avg_disc, count_order""",
    2: """Minimum cost supplier. For each part of size 15 whose type ends in 'BRASS', find the suppliers in the 'EUROPE' region who supply it at the
minimum supply cost among all European suppliers of that part. Report the supplier's account balance, supplier name, nation name, part key,
part manufacturer, supplier address, supplier phone, and supplier comment.
Sort by account balance descending, then nation name, supplier name, and part key. Return the top 100 rows.
Output columns: s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment""",
    3: """Shipping priority. For customers in market segment 'BUILDING', find orders placed before 1995-03-15 that had line items shipped after 1995-03-15.
For each such order, report the order key, the revenue (sum of extended price * (1 - discount) over those line items shipped after 1995-03-15),
the order date, and the ship priority.
Sort by revenue descending, then order date. Return the top 10 rows.
Output columns: l_orderkey, revenue, o_orderdate, o_shippriority""",
    4: """Order priority checking. Count the orders placed in the quarter starting 1993-07-01 (on or after 1993-07-01 and before 1993-10-01) that have at least
one line item received later than its commit date. Report the count per order priority.
Sort by order priority.
Output columns: o_orderpriority, order_count""",
    5: """Local supplier volume. For each nation in the 'ASIA' region, compute the revenue (sum of extended price * (1 - discount)) from line items where both the
customer and the supplier are in that same nation, for orders placed in 1994 (on or after 1994-01-01 and before 1995-01-01).
Sort by revenue descending.
Output columns: n_name, revenue""",
    6: """Forecasting revenue change. Compute the total of extended price * discount for line items shipped in 1994 (on or after 1994-01-01 and before 1995-01-01)
with a discount between 0.05 and 0.07 inclusive and a quantity less than 24.
Output columns: revenue""",
    7: """Volume shipping. Report the value of goods shipped between 'FRANCE' and 'GERMANY' (supplier in one nation, customer in the other, both directions)
for line items shipped between 1995-01-01 and 1996-12-31 inclusive. Value is extended price * (1 - discount).
Group by supplier nation name, customer nation name, and ship year.
Sort by supplier nation, customer nation, year.
Output columns: supp_nation, cust_nation, l_year, revenue""",
    8: """National market share. For orders placed between 1995-01-01 and 1996-12-31 inclusive by customers in the 'AMERICA' region, considering only parts of type
'ECONOMY ANODIZED STEEL', compute for each order year the fraction of revenue (extended price * (1 - discount)) supplied by suppliers from 'BRAZIL'
out of the total revenue for those line items (suppliers from any nation).
Sort by year.
Output columns: o_year, mkt_share""",
    9: """Product type profit measure. For parts whose name contains 'green', compute the profit per supplier nation and order year, where profit for a line item
is extended price * (1 - discount) - (the supplier's supply cost for that part from partsupp) * quantity.
Sort by nation name ascending, then year descending.
Output columns: nation, o_year, sum_profit""",
    10: """Returned item reporting. For orders placed in the quarter starting 1993-10-01 (on or after 1993-10-01 and before 1994-01-01), find the customers with returned
line items (return flag 'R'). For each customer report: customer key, customer name, lost revenue (sum of extended price * (1 - discount) over those returned items),
account balance, nation name, address, phone, and comment.
Sort by revenue descending. Return the top 20 rows.
Output columns: c_custkey, c_name, revenue, c_acctbal, n_name, c_address, c_phone, c_comment""",
    11: """Important stock identification. For suppliers in 'GERMANY', compute for each part the total value of available stock (supply cost * available quantity, summed
over those suppliers). Report only parts whose value is greater than 0.0001 times the total value of all available stock from German suppliers.
Sort by value descending.
Output columns: ps_partkey, value""",
    12: """Shipping modes and order priority. For line items with ship mode 'MAIL' or 'SHIP' that were received in 1994 (on or after 1994-01-01 and before 1995-01-01),
were received after their commit date, and were shipped before their commit date: count, per ship mode, the number of such line items belonging to orders with
priority '1-URGENT' or '2-HIGH' (high_line_count) and the number belonging to orders with any other priority (low_line_count).
Sort by ship mode.
Output columns: l_shipmode, high_line_count, low_line_count""",
    13: """Customer distribution. For every customer (including those with no orders), count their orders, excluding orders whose comment matches the pattern
'%special%requests%'. Then report how many customers have each order count.
Sort by number of customers descending, then order count descending.
Output columns: c_count, custdist""",
    14: """Promotion effect. For line items shipped in September 1995 (on or after 1995-09-01 and before 1995-10-01), compute the percentage (0-100) of revenue
(extended price * (1 - discount)) that came from parts whose type starts with 'PROMO'.
Output columns: promo_revenue""",
    15: """Top supplier. Compute each supplier's total revenue (extended price * (1 - discount)) from line items shipped in the quarter starting 1996-01-01
(on or after 1996-01-01 and before 1996-04-01). Report the supplier(s) with the maximum total revenue: supplier key, name, address, phone, and total revenue.
Sort by supplier key.
Output columns: s_suppkey, s_name, s_address, s_phone, total_revenue""",
    16: """Parts/supplier relationship. Count the number of distinct suppliers who can supply parts that are not of brand 'Brand#45', whose type does not start with
'MEDIUM POLISHED', and whose size is one of 49, 14, 23, 45, 19, 3, 36, 9. Exclude suppliers whose comment matches '%Customer%Complaints%'.
Group by brand, type, and size.
Sort by supplier count descending, then brand, type, size ascending.
Output columns: p_brand, p_type, p_size, supplier_cnt""",
    17: """Small-quantity-order revenue. For parts of brand 'Brand#23' with container 'MED BOX', consider line items whose quantity is less than 20% of the average
quantity of all line items for that same part. Report the sum of extended price of those line items divided by 7.0 (average yearly revenue).
Output columns: avg_yearly""",
    18: """Large volume customer. Find orders whose total line item quantity is greater than 300. For each, report customer name, customer key, order key, order date,
order total price, and the total quantity of the order.
Sort by total price descending, then order date. Return the top 100 rows.
Output columns: c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum_quantity""",
    19: """Discounted revenue. Compute total revenue (extended price * (1 - discount)) for line items with ship mode 'AIR' or 'AIR REG' and ship instruction 'DELIVER IN PERSON'
whose part satisfies one of:
 (a) brand 'Brand#12', container in ('SM CASE','SM BOX','SM PACK','SM PKG'), quantity between 1 and 11 inclusive, size between 1 and 5 inclusive;
 (b) brand 'Brand#23', container in ('MED BAG','MED BOX','MED PKG','MED PACK'), quantity between 10 and 20 inclusive, size between 1 and 10 inclusive;
 (c) brand 'Brand#34', container in ('LG CASE','LG BOX','LG PACK','LG PKG'), quantity between 20 and 30 inclusive, size between 1 and 15 inclusive.
Output columns: revenue""",
    20: """Potential part promotion. Find suppliers in 'CANADA' who have, for at least one part whose name starts with 'forest', an available quantity greater than half
of the total quantity of that part shipped by that supplier in 1994 (shipped on or after 1994-01-01 and before 1995-01-01).
Report supplier name and address.
Sort by supplier name.
Output columns: s_name, s_address""",
    21: """Suppliers who kept orders waiting. For suppliers in 'SAUDI ARABIA', count line items where: the order has status 'F'; the supplier's line item was received
after its commit date; the order has at least one line item from a different supplier; and no line item from a different supplier on that order was received
after its commit date. Report supplier name and the count.
Sort by count descending, then supplier name. Return the top 100 rows.
Output columns: s_name, numwait""",
    22: """Global sales opportunity. Consider customers whose country code (the first two characters of their phone number) is one of '13','31','23','29','30','18','17',
whose account balance is greater than the average positive account balance (balance > 0.00) of customers with those country codes, and who have never placed an order.
For each country code report the number of such customers and the sum of their account balances.
Sort by country code.
Output columns: cntrycode, numcust, totacctbal""",
}
